#!/usr/bin/env python3
"""
从音频文件生成美观的波形可视化，并保存为 MP4 文件。
"""
import argparse
import json
import math
import subprocess as sp
import sys
import tempfile
from pathlib import Path
import datetime  # 新增：用于获取当前日期

import cairo
import numpy as np
import tqdm

_is_main = False


def colorize(text, color):
    """
    用 ANSI `color` 代码包裹 `text` 文本，实现终端颜色显示。参考：
    https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences
    """
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def fatal(msg):
    """
    处理严重错误。若当前模块不是主模块（__main__），则不执行任何操作；
    若是主模块，则显示错误信息并终止程序。
    """
    if _is_main:
        head = "错误："
        if sys.stderr.isatty():
            head = colorize("错误：", 1)
        print(head + str(msg), file=sys.stderr)
        sys.exit(1)


def read_info(media):
    """
    获取媒体文件的相关信息并返回。
    """
    proc = sp.run([
        'ffprobe', "-loglevel", "panic",
        str(media), '-print_format', 'json', '-show_format', '-show_streams'
    ],
                  capture_output=True)
    if proc.returncode != 0:
        raise IOError(f"{media}不存在或格式错误。")
    return json.loads(proc.stdout.decode('utf-8'))


def read_audio(audio, seek=None, duration=None):
    """
    读取音频文件内容。从 `seek`（默认0）秒开始，读取 `duration`（默认全部）秒的音频。
    返回格式：`float[声道数, 采样点数]`。
    """

    info = read_info(audio)
    channels = None
    stream = info['streams'][0]
    if stream["codec_type"] != "audio":
        raise ValueError(f"{audio}应仅包含音频内容。")
    channels = stream['channels']
    samplerate = float(stream['sample_rate'])

    # 使用 ffmpeg 读取音频
    command = ['ffmpeg', '-y']
    command += ['-loglevel', 'panic']
    if seek is not None:
        command += ['-ss', str(seek)]
    command += ['-i', audio]
    if duration is not None:
        command += ['-t', str(duration)]
    command += ['-f', 'f32le']
    command += ['-']

    proc = sp.run(command, check=True, capture_output=True)
    wav = np.frombuffer(proc.stdout, dtype=np.float32)
    return wav.reshape(-1, channels).T, samplerate


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def envelope(wav, window, stride):
    """
    提取波形 `wav`（float[采样点数]）的包络线。
    使用平均池化算法，池化窗口大小为 `window` 个采样点，步长为 `stride`。
    """
    wav = np.pad(wav, window // 2)
    out = []
    for off in range(0, len(wav) - window, stride):
        frame = wav[off:off + window]
        out.append(np.maximum(frame, 0).mean())
    out = np.array(out)
    # 基于 sigmoid 函数的简易音频压缩处理
    out = 1.9 * (sigmoid(2.5 * out) - 0.5)
    return out


def draw_env(envs, out, fg_colors, bg_color, size):
    """
    内部绘图函数：使用 Cairo 绘制单帧波形图（立体声时为两帧），并以 PNG 格式保存到 `out` 文件。
    参数说明：
    - envs：各声道的包络线列表，每个包络线为 float[柱形数量]，代表需绘制的柱形高度
    - 每个数值对应一个可视化柱形
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, *size)
    ctx = cairo.Context(surface)
    ctx.scale(*size)  # 缩放坐标系至 [0,1] 范围

    # 绘制背景
    ctx.set_source_rgb(*bg_color)
    ctx.rectangle(0, 0, 1, 1)
    ctx.fill()

    K = len(envs)  # 需绘制的波形数量（垂直堆叠）
    T = len(envs[0])  # 时间步长数量（柱形总数）
    pad_ratio = 0.1  # 柱形之间的间距比例
    width = 1. / (T * (1 + 2 * pad_ratio))  # 单个柱形宽度
    pad = pad_ratio * width  # 柱形两侧间距
    delta = 2 * pad + width  # 相邻柱形的中心间距

    ctx.set_line_width(width)
    for step in range(T):  # 遍历每个时间步长（每个柱形）
        for i in range(K):  # 遍历每个波形（声道）
            half = 0.5 * envs[i][step]  # 柱形的（半）高度
            half /= K  # 按波形数量均分垂直空间
            midrule = (1 + 2 * i) / (2 * K)  # 第 i 个波形的中线位置
            
            # 绘制柱形上半部分（不透明）
            ctx.set_source_rgb(*fg_colors[i])
            ctx.move_to(pad + step * delta, midrule - half)
            ctx.line_to(pad + step * delta, midrule)
            ctx.stroke()
            
            # 绘制柱形下半部分（半透明）
            ctx.set_source_rgba(*fg_colors[i], 0.8)
            ctx.move_to(pad + step * delta, midrule)
            ctx.line_to(pad + step * delta, midrule + 0.9 * half)
            ctx.stroke()

    surface.write_to_png(out)


def interpole(x1, y1, x2, y2, x):
    """线性插值函数：根据两点 (x1,y1)、(x2,y2) 计算 x 对应的 y 值"""
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def visualize(audio,
              tmp,
              out,
              seek=None,
              duration=None,
              rate=60,
              bars=50,
              speed=4,
              time=0.4,
              oversample=3,
              fg_color=(.2, .2, .2),
              fg_color2=(.5, .3, .6),
              bg_color=(1, 1, 1),
              size=(400, 400),
              stereo=False,
              ):
    """
    生成音频文件的波形可视化动画。
    参数说明：
    - audio：输入音频文件路径
    - tmp：临时文件夹路径（用于存储中间帧）
    - out：输出视频文件路径
    - seek：音频起始读取位置（秒），默认从开头开始
    - duration：读取音频的时长（秒），默认读取全部
    - rate：输出视频的帧率，默认60
    - bars：单帧画面中显示的柱形数量，默认50
    - speed：波形过渡的基础速度，实际速度会根据音量在基础速度的0.5-2倍之间变化，默认4
    - time：单帧画面显示的音频时长（秒），默认0.4
    - oversample：过采样率，值越高波形变化越频繁，默认3
    - fg_color：前景色（柱形颜色），RGB格式，默认(0.2, 0.2, 0.2)
    - fg_color2：立体声模式下第二个波形的颜色，RGB格式，默认(0.5, 0.3, 0.6)
    - bg_color：背景色，RGB格式，默认(1, 1, 1)（白色）
    - size：输出视频的分辨率（宽, 高），默认(400, 400)
    - stereo：是否启用立体声模式（显示两个波形），默认False
    """
    try:
        wav, sr = read_audio(audio, seek=seek, duration=duration)
    except (IOError, ValueError) as err:
        fatal(err)
        raise
    
    # 按声道拆分波形（立体声/单声道处理）
    wavs = []
    if stereo:
        assert wav.shape[0] == 2, '启用立体声模式需要输入立体声音频文件'
        wavs.append(wav[0])  # 左声道
        wavs.append(wav[1])  # 右声道
    else:
        wav = wav.mean(0)  # 多声道转为单声道（取平均值）
        wavs.append(wav)

    # 对各声道波形进行标准化处理（标准差归一化）
    for i, wav in enumerate(wavs):
        wavs[i] = wav / wav.std()

    # 计算池化窗口大小和步长
    window = int(sr * time / bars)  # 每个柱形对应的采样点数量
    stride = int(window / oversample)  # 池化步长

    # 提取各声道的波形包络线
    envs = []
    for wav in wavs:
        env = envelope(wav, window, stride)
        env = np.pad(env, (bars // 2, 2 * bars))  # 前后补零，避免边缘截断
        envs.append(env)

    # 计算总帧数并生成汉宁窗（用于柱形平滑过渡）
    duration = len(wavs[0]) / sr  # 实际处理的音频时长
    frames = int(rate * duration)  # 总帧数
    smooth = np.hanning(bars)  # 汉宁窗（用于柱形幅度平滑）

    print("正在生成帧...")
    # 遍历生成每一帧
    for idx in tqdm.tqdm(range(frames), unit=" 帧", ncols=80):
        # 计算当前帧对应的包络线位置
        pos = (((idx / rate)) * sr) / stride / bars
        off = int(pos)  # 整数部分（当前包络段起始索引）
        loc = pos - off  # 小数部分（用于两段包络线的插值）

        # 对每个声道的包络线进行插值平滑
        denvs = []
        for env in envs:
            # 获取当前段和下一段包络线
            env1 = env[off * bars : (off + 1) * bars]
            env2 = env[(off + 1) * bars : (off + 2) * bars]

            # 根据音量调整过渡速度（音量越大，过渡越快）
            maxvol = math.log10(1e-4 + env2.max()) * 10  # 计算最大音量（dB）
            speedup = np.clip(interpole(-6, 0.5, 0, 2, maxvol), 0.5, 2)  # 速度调整系数
            w = sigmoid(speed * speedup * (loc - 0.5))  # 插值权重（sigmoid平滑）
            
            # 两段包络线插值融合，并应用汉宁窗平滑
            denv = (1 - w) * env1 + w * env2
            denv *= smooth
            denvs.append(denv)
        
        # 绘制当前帧并保存为PNG
        draw_env(denvs, tmp / f"{idx:06d}.png", (fg_color, fg_color2), bg_color, size)

    # 构建FFmpeg音频处理命令（用于将音频嵌入视频）
    audio_cmd = []
    if seek is not None:
        audio_cmd += ["-ss", str(seek)]
    audio_cmd += ["-i", audio.resolve()]
    if duration is not None:
        audio_cmd += ["-t", str(duration)]

    print("正在编码动画视频... ")
    # 参考：https://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/
    sp.run([
        "ffmpeg", "-y", "-loglevel", "panic", "-r",
        str(rate), "-f", "image2", "-s", f"{size[0]}x{size[1]}", "-i", "%06d.png"
    ] + audio_cmd + [
        "-c:a", "aac", "-vcodec", "libx264", "-crf", "10", "-pix_fmt", "yuv420p",
        out.resolve()
    ],
           check=True,
           cwd=tmp)


def parse_color(colorstr):
    """
    将逗号分隔的RGB颜色字符串转换为浮点元组。
    输入格式示例："0.5,0.3,0.8"，输出格式：(0.5, 0.3, 0.8)
    """
    try:
        r, g, b = [float(i) for i in colorstr.split(",")]
        return r, g, b
    except ValueError:
        fatal("颜色格式错误！正确格式为3个浮点数，用逗号分隔（如0.xx,0.xx,0.xx），按RGB顺序输入。")
        raise


def main():
    # 获取当前日期并格式化为YYYYMMDD
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        'SeeVoice', description="从音频文件生成美观的MP4波形动画。")
    parser.add_argument("-r", "--rate", type=int, default=60, help="视频帧率，默认60。")
    parser.add_argument("--stereo", action='store_true',
                        help="为立体声文件生成两个独立波形（左/右声道）。")
    parser.add_argument("-c",
                        "--color",
                        default=[0.03, 0.6, 0.3],
                        type=parse_color,
                        dest="color",
                        help="柱形颜色，格式为'r,g,b'（每个值范围0-1），默认0.03,0.6,0.3。")
    parser.add_argument("-c2",
                        "--color2",
                        default=[0.5, 0.3, 0.6],
                        type=parse_color,
                        dest="color2",
                        help="立体声模式下第二个波形的颜色，格式同--color，默认0.5,0.3,0.6。")
    parser.add_argument("--white", action="store_true",
                        help="使用白色背景（默认使用黑色背景）。")
    parser.add_argument("-B",
                        "--bars",
                        type=int,
                        default=50,
                        help="单帧画面中显示的柱形数量，默认50。")
    parser.add_argument("-O", "--oversample", type=float, default=4,
                        help="过采样率（值越低波形变化越平缓，越高越灵敏），默认4。")
    parser.add_argument("-T", "--time", type=float, default=0.4,
                        help="单帧画面显示的音频时长（秒），默认0.4。")
    parser.add_argument("-S", "--speed", type=float, default=4,
                        help="波形过渡基础速度（值越高过渡越快），默认4。")
    parser.add_argument("-W",
                        "--width",
                        type=int,
                        default=480,
                        help="输出视频的宽度（像素），默认480。")
    parser.add_argument("-H",
                        "--height",
                        type=int,
                        default=300,
                        help="输出视频的高度（像素），默认300。")
    parser.add_argument("-s", "--seek", type=float, help="音频起始处理位置（秒），默认从开头开始。")
    parser.add_argument("-d", "--duration", type=float, help="音频处理时长（秒），默认处理全部音频。")
    parser.add_argument("audio", type=Path, help='输入音频文件的路径。')
    parser.add_argument("out",
                        type=Path,
                        nargs='?',
                        # 修改默认输出文件名为out_当前日期.mp4
                        default=Path(f'{current_date}.mp4'),
                        help=f'输出视频文件的路径，默认值为./{current_date}.mp4。')
    args = parser.parse_args()

    # 创建临时文件夹并执行可视化生成
    with tempfile.TemporaryDirectory() as tmp:
        visualize(args.audio,
                  Path(tmp),
                  args.out,
                  seek=args.seek,
                  duration=args.duration,
                  rate=args.rate,
                  bars=args.bars,
                  speed=args.speed,
                  oversample=args.oversample,
                  time=args.time,
                  fg_color=args.color,
                  fg_color2=args.color2,
                  bg_color=[1.0 if args.white else 0.0] * 3,  # 根据--white参数设置背景色
                  size=(args.width, args.height),
                  stereo=args.stereo)


if __name__ == "__main__":
    _is_main = True
    main()
