# SeeWav：音频波形动画生成工具

SeeWav可以为您的音频波形生成精美的动画。
查看演示请点击下方图片：

<p align="center">
<a href="https://ai.honu.io/misc/seewav.mp4">
<img src="./seewav.png" alt="seewav演示"></a></p>

## 安装

您需要Python 3.7版本。
您需要安装`ffmpeg`，并确保其支持`libx264`和`aac`编解码器。
在使用Homebrew的Mac OS X系统上，运行`brew install ffmpeg`；在Ubuntu系统上，运行`sudo apt-get install ffmpeg`。
如果使用Anaconda，也可以运行`conda install -c conda-forge ffmpeg`。


```bash
pip3 install seewav
```

## 使用方法


```bash
seewav 音频文件 [输出文件]
```
默认情况下，输出文件为`out.mp4`。可用选项如下：

```bash
usage: seewav [-h] [-r 帧率] [--stereo] [-c 颜色] [-c2 颜色2] [--white]
              [-B 柱形数量] [-O 过采样率] [-T 单帧时长] [-S 过渡速度] [-W 宽度]
              [-H 高度] [-s 起始时间] [-d 持续时长]
              audio [out]

从音频文件生成精美的MP4动画。

位置参数:
  audio                 音频文件路径
  out                   输出文件路径，默认值为./out.mp4

可选参数:
  -h, --help            显示此帮助信息并退出
  -r 帧率, --rate 帧率  视频帧率。
  --stereo              为立体声文件生成两个波形。
  -c 颜色, --color 颜色  柱形颜色，格式为'r,g,b'（每个值范围0-1）。
  -c2 颜色2, --color2 颜色2
                        立体声模式下第二个波形的颜色，格式为'r,g,b'（每个值范围0-1）。
  --white               使用白色背景，默认使用黑色背景。
  -B 柱形数量, --bars 柱形数量
                        单帧画面中显示的柱形数量
  -O 过采样率, --oversample 过采样率
                        值越低，波形变化越平缓。
  -T 单帧时长, --time 单帧时长
                        单帧画面显示的音频时长（秒）。
  -S 过渡速度, --speed 过渡速度
                        值越高，帧之间的过渡越快。
  -W 宽度, --width 宽度  动画的宽度（像素）
  -H 高度, --height 高度  动画的高度（像素）
  -s 起始时间, --seek 起始时间
                        音频处理的起始时间（秒）。
  -d 持续时长, --duration 持续时长
                        从起始时间开始的处理时长（秒）。
```