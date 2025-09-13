#!/usr/bin/env python3

from pathlib import Path

from setuptools import setup

# 项目基础配置
NAME = 'SeeVoice'  # 包名称（需与PyPI仓库名称一致）
DESCRIPTION = '音频波形可视化工具'  # 包的简短描述
URL = 'https://github.com/adefossez/SeeVoice'  # 项目仓库地址
EMAIL = 'xxx@gmail.com'  # 作者邮箱（可根据实际情况修改）
AUTHOR = 'xxx'  # 作者名称（可根据实际情况修改）
REQUIRES_PYTHON = '>=3.6.0'  # 项目所需的最低Python版本
VERSION = "0.1.1a1"  # 项目版本号（遵循语义化版本规范，a表示alpha测试版）

# 获取当前文件所在目录路径
HERE = Path(__file__).parent

# 从requirements.txt文件读取项目依赖列表（过滤空行）
REQUIRED = [i.strip() for i in open("requirements.txt") if i.strip()]

# 尝试读取README.md作为长描述（若文件不存在则使用短描述）
try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# 包的安装配置
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',  # 长描述的格式（Markdown）
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    py_modules=['SeeVoice'],  # 指定需包含的Python模块（对应SeeVoice.py文件）
    # 配置命令行入口：安装后可直接通过"SeeVoice"命令调用SeeVoice模块的main函数
    entry_points={
        'console_scripts': ['SeeVoice=SeeVoice:main'],
    },
    install_requires=REQUIRED,  # 项目依赖的第三方包列表
    include_package_data=True,  # 是否包含包内非代码文件（如README、LICENSE等）
    license='无版权许可协议（Unlicense）',  # 许可协议类型（Unlicense为无版权协议）
    # 包的分类信息（遵循PyPI Trove分类器标准，用于PyPI平台检索分类）
    classifiers=[
        # 完整分类器列表：https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Topic :: Multimedia :: Sound/Audio',  # 多媒体 -> 声音/音频
        'Topic :: Multimedia :: Video',  # 多媒体 -> 视频
    ],
)