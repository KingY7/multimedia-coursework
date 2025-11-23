# 2022217157-黎耀-《多媒体技术》实验

本仓库包含《多媒体技术》课程的实验项目：**PCA与2D-PCA特征脸识别**、**基于傅里叶描述子的数字识别** 以及 **视频制作**。

## ? 项目结构

```
2022217157-黎耀-《多媒体技术》实验/
├── test1/                       # 实验一：PCA与2D-PCA特征脸识别
│   ├── pca_1d.py                # 1D-PCA算法实现
│   ├── pca_2d.py                # 2D-PCA算法实现
│   ├── visualize_pca.py         # 可视化分析脚本
│   ├── requirements.txt         # 实验一依赖
│   └── README.md                # 实验一详细说明
│
├── test2/                       # 实验二：基于傅里叶描述子的数字识别
│   ├── fourier_descriptor.py    # 傅里叶描述子核心实现
│   ├── digit_recognition.py     # 数字识别系统
│   ├── realtime_digit_recognition.py # 实时识别脚本
│   ├── conda环境配置.txt        # 环境配置说明
│   └── README.md                # 实验二详细说明
│
├── 实验三视频制作.mp4           # 实验三：视频制作成果
└── README.md                    # 项目总说明文档
```

---

## ? 实验一：PCA与2D-PCA特征脸识别

本实验旨在深入理解并对比一维主成分分析（1D-PCA）和二维主成分分析（2D-PCA）在人脸识别中的应用。

- **核心内容**：
  - 实现 1D-PCA 和 2D-PCA 算法。
  - 使用 LFW (Labeled Faces in the Wild) 数据集进行训练和测试。
  - 对比两种方法在识别准确率、训练时间、特征维度等方面的性能。
  - 可视化特征脸（Eigenfaces）。

- **快速开始**：
  ```bash
  cd test1
  pip install -r requirements.txt
  python pca_1d.py  # 运行1D-PCA
  python pca_2d.py  # 运行2D-PCA
  ```

> 更多详情请查看 [test1/README.md](test1/README.md)

---

## ? 实验二：基于傅里叶描述子的数字识别

本实验探讨如何利用傅里叶描述子（Fourier Descriptors）提取图像形状特征，并应用于手写数字识别。

- **核心内容**：
  - 提取图像轮廓并计算傅里叶描述子。
  - 实现具有旋转、缩放、平移不变性的形状特征提取。
  - 使用 sklearn Digits 数据集进行数字识别。
  - 对比不同分类器（KNN, SVM等）的识别效果。

- **快速开始**：
  ```bash
  cd test2
  # 建议参考 'conda环境配置.txt' 创建独立环境
  pip install numpy scipy opencv-python scikit-learn scikit-image matplotlib seaborn pillow tqdm
  python fourier_descriptor.py  # 测试傅里叶描述子
  python digit_recognition.py   # 运行数字识别
  ```

> 更多详情请查看 [test2/README.md](test2/README.md)

---

## ? 实验三：视频制作

实验三为视频制作任务，成果文件为 `实验三视频制作.mp4`。

**说明：** 在视频制作过程中，使用AI工具生成字幕时，部分文字可能出现乱码。尽管多次尝试重新生成，这个问题仍然存在，属于目前AI技术的局限性。

---

## ?? 环境配置

建议使用 Anaconda 创建统一的 Python 环境来运行两个实验。

```bash
# 创建环境 (Python 3.9)
conda create -n multimedia_lab python=3.9 -y

# 激活环境
conda activate multimedia_lab

# 安装通用依赖
pip install numpy scipy matplotlib seaborn scikit-learn pillow tqdm opencv-python scikit-image
```

## ? 作者信息

- **姓名**：黎耀
- **学号**：2022217157
- **课程**：《多媒体技术》