# spectool

**spectool** 是一个用于光谱数据处理的工具包，提供了一系列常见的光谱数据减缩函数，涵盖了光谱数据的重采样、平滑、归一化、径向速度测量等功能。

## 主要功能模块

- **光谱重采样与插值**: 提供了重采样函数，如根据给定波长范围对光谱进行重采样。
- **光谱平滑与滤波**: 提供了基于高斯滤波器、rotation kernel滤波器等多种平滑方法，包括对波长和速度的平滑。
- **光谱归一化**: 支持使用高斯滤波、分段多项式等方法对光谱进行归一化处理。
- **光谱匹配与拟合**: 实现了对不同光谱间的匹配，使用多项式或其他拟合方法对光谱进行缩放与调整。
- **径向速度测量**: 包含了使用光谱数据计算径向速度的函数。

## 安装

可以通过以下命令直接安装：

```bash
pip install git+https://github.com/zhang-zhixiang/spectool.git
```

或下载源代码并安装：

```bash
git clone https://github.com/zhang-zhixiang/spectool.git
cd spectool
python setup.py install
```

## 依赖

该项目依赖以下库：

- GSL (>= 2.7)
- FFTW3
- pybind11
- PyAstronomy

安装方法请参见[README](https://github.com/zhang-zhixiang/spectool/blob/master/README.md)。