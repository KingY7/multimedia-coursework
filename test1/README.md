# 实验一：PCA与2D-PCA特征脸识别

## 实验目的

1. 深入理解一维主成分分析（1D-PCA）和二维主成分分析（2D-PCA）算法原理
2. 掌握PCA在人脸识别中的应用方法
3. 对比分析1D-PCA和2D-PCA的性能差异
4. 学习特征脸（Eigenface）识别技术

## 实验内容

### 1. 一维主成分分析（1D-PCA）

**算法原理：**
- 将图像展开为一维向量
- 计算所有训练样本的协方差矩阵
- 对协方差矩阵进行特征值分解
- 选择前k个最大特征值对应的特征向量作为主成分
- 将图像投影到主成分空间进行降维

**优点：**
- 算法成熟，理论完善
- 适用于各种数据类型
- 降维效果明显

**缺点：**
- 破坏了图像的二维结构信息
- 协方差矩阵维度高，计算复杂度大
- 需要较多的主成分才能保持良好的识别率

### 2. 二维主成分分析（2D-PCA）

**算法原理：**
- 直接利用图像的二维矩阵形式
- 计算图像协方差矩阵 G = (1/n) Σ A_i^T * A_i
- 对G进行特征值分解得到投影矩阵
- 将图像矩阵投影到低维空间：Y = A * X
- 保留图像的空间结构信息

**优点：**
- 协方差矩阵维度较小，计算效率高
- 保留了图像的二维结构信息
- 需要较少的主成分即可达到良好效果

**缺点：**
- 特征维度相对1D-PCA可能更高
- 理论研究相对较少

## 项目结构

```
实验一/
├── pca_1d.py                    # 1D-PCA算法实现
├── pca_2d.py                    # 2D-PCA算法实现
├── compare_pca.py               # 对比实验脚本
├── visualize_pca.py             # 可视化分析脚本
├── requirements.txt             # 依赖包列表
├── README.md                    # 实验说明文档
└── results/                     # 实验结果目录
    ├── eigenfaces_1d.png        # 1D特征脸
    ├── eigenfaces_2d.png        # 2D特征脸
    ├── comparison_results.png   # 综合对比图
    └── comparison_report.txt    # 对比报告
```

## 环境配置

### 依赖库

- Python >= 3.7
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0
- scikit-learn >= 0.24.0
- seaborn >= 0.11.0

### 安装方法

```bash
pip install -r requirements.txt
```

## 使用说明

### 1. 运行1D-PCA实验

```bash
python pca_1d.py
```

**输出结果：**
- 特征脸可视化图（eigenfaces_1d.png）
- 累积解释方差比图（explained_variance_1d.png）
- 混淆矩阵（confusion_matrix_1d.png）
- 识别准确率和分类报告

### 2. 运行2D-PCA实验

```bash
python pca_2d.py
```

**输出结果：**
- 2D特征脸可视化图（eigenfaces_2d.png）
- 累积解释方差比图（explained_variance_2d.png）
- 混淆矩阵（confusion_matrix_2d.png）
- 识别准确率和分类报告

### 3. 运行对比实验

```bash
python compare_pca.py
```

**输出结果：**
- 综合对比图表（comparison_results.png）
  - 训练时间对比
  - 推理时间对比
  - 性能指标对比（准确率、精确率、召回率、F1分数）
  - 特征维度对比
- 图像重构效果对比（reconstruction_comparison.png）
- 详细对比报告（comparison_report.txt）

### 4. 运行可视化分析

```bash
python visualize_pca.py
```

**输出结果：**
- 不同主成分数量下的性能曲线
- 特征空间可视化（t-SNE降维）
- 更多详细分析图表

## 数据集说明

本实验使用 **LFW（Labeled Faces in the Wild）** 人脸数据集：

- **数据来源：** scikit-learn内置数据集
- **样本数量：** 筛选每人至少70张照片的人物
- **图像尺寸：** 50 × 37 像素（resize=0.4）
- **类别数量：** 7个人物
- **数据划分：** 75%训练集，25%测试集

## 实验参数设置

### 1D-PCA
- 主成分数量：100
- 分类器：最近邻分类器（Nearest Neighbor）

### 2D-PCA
- 主成分数量：50
- 图像尺寸：50 × 37
- 分类器：最近邻分类器（Nearest Neighbor）

## 评价指标

1. **准确率（Accuracy）：** 正确预测的样本比例
2. **精确率（Precision）：** 预测为正例中实际为正例的比例
3. **召回率（Recall）：** 实际正例中被正确预测的比例
4. **F1分数（F1-Score）：** 精确率和召回率的调和平均值
5. **训练时间：** 模型训练所需时间
6. **推理时间：** 测试集预测所需时间
7. **特征维度：** 降维后的特征空间维度

## 实验结果分析

### 预期结果

1. **识别准确率：** 两种方法都能达到70%以上的识别准确率
2. **计算效率：** 2D-PCA在训练和推理速度上可能更快
3. **特征维度：** 2D-PCA的特征维度相对较高
4. **降维效果：** 两种方法都能显著降低特征维度

### 对比分析要点

- **算法复杂度：** 2D-PCA协方差矩阵维度更小
- **特征保留：** 2D-PCA保留图像二维结构信息
- **识别性能：** 在不同参数设置下性能各有优劣
- **应用场景：** 根据具体需求选择合适的方法

## 扩展实验

1. **参数优化：** 调整主成分数量，观察性能变化
2. **特征融合：** 结合1D-PCA和2D-PCA的特征
3. **分类器对比：** 使用SVM、神经网络等不同分类器
4. **数据增强：** 应用图像增强技术提升识别率
5. **实时识别：** 优化算法实现实时人脸识别

## 常见问题

### Q1: 运行时提示找不到数据集怎么办？

**A:** 首次运行会自动下载LFW数据集，需要联网。如果下载失败，可以手动下载数据集并放置到scikit-learn数据目录。

### Q2: 为什么2D-PCA的特征维度反而更高？

**A:** 2D-PCA保留了图像的高度维度，特征形状为 (height × n_components)，虽然n_components较小，但总维度可能大于1D-PCA。

### Q3: 如何提高识别准确率？

**A:** 
- 增加主成分数量
- 使用更复杂的分类器（如SVM）
- 增加训练样本数量
- 进行数据预处理和增强

### Q4: 内存不足怎么办？

**A:** 
- 减少数据集规模（调整min_faces_per_person参数）
- 降低图像分辨率（调整resize参数）
- 减少主成分数量

## 参考文献

1. Turk, M., & Pentland, A. (1991). Eigenfaces for recognition. *Journal of cognitive neuroscience*, 3(1), 71-86.

2. Yang, J., Zhang, D., Frangi, A. F., & Yang, J. Y. (2004). Two-dimensional PCA: a new approach to appearance-based face representation and recognition. *IEEE transactions on pattern analysis and machine intelligence*, 26(1), 131-137.

3. Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review and recent developments. *Philosophical Transactions of the Royal Society A*, 374(2065), 20150202.

## 实验报告要求

1. **实验原理：** 详细阐述1D-PCA和2D-PCA的算法原理
2. **实验步骤：** 记录实验过程和关键代码
3. **实验结果：** 展示所有生成的图表和数据
4. **对比分析：** 深入分析两种方法的优缺点
5. **实验总结：** 总结实验收获和改进方向

## 作者信息

- **实验名称：** PCA与2D-PCA特征脸识别
- **课程：** 多媒体技术
- **实验编号：** 实验一

## 许可证

本实验代码仅供学习和研究使用。
