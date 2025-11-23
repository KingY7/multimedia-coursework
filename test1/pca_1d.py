"""
一维主成分分析（1D-PCA）算法实现
用于特征脸识别
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class PCA1D:
    """一维主成分分析类"""
    
    def __init__(self, n_components=None):
        """
        初始化PCA1D
        
        Args:
            n_components: 主成分数量，如果为None则保留所有成分
        """
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None
        
    def fit(self, X):
        """
        训练PCA模型
        
        Args:
            X: 输入数据，形状为(n_samples, n_features)
        """
        # 计算均值
        self.mean_ = np.mean(X, axis=0)
        
        # 中心化数据
        X_centered = X - self.mean_
        
        # 计算协方差矩阵
        cov_matrix = np.cov(X_centered.T)
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 按特征值降序排列
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 选择主成分数量
        if self.n_components is None:
            self.n_components = len(eigenvalues)
        else:
            self.n_components = min(self.n_components, len(eigenvalues))
            
        # 保存主成分和解释方差比
        self.components_ = eigenvectors[:, :self.n_components]
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_variance
        
        return self
    
    def transform(self, X):
        """
        将数据投影到主成分空间
        
        Args:
            X: 输入数据
            
        Returns:
            投影后的数据
        """
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)
    
    def inverse_transform(self, X_transformed):
        """
        将投影数据还原到原始空间
        
        Args:
            X_transformed: 投影后的数据
            
        Returns:
            还原后的数据
        """
        return np.dot(X_transformed, self.components_.T) + self.mean_


class EigenfaceRecognizer:
    """基于1D-PCA的特征脸识别器"""
    
    def __init__(self, n_components=100):
        """
        初始化特征脸识别器
        
        Args:
            n_components: 主成分数量
        """
        self.pca = PCA1D(n_components=n_components)
        self.labels = None
        self.label_encoder = None
        
    def fit(self, X, y):
        """
        训练识别器
        
        Args:
            X: 训练图像数据，形状为(n_samples, n_features)
            y: 标签
        """
        # 训练PCA
        self.pca.fit(X)
        
        # 保存标签信息
        self.labels = np.unique(y)
        self.label_encoder = {label: idx for idx, label in enumerate(self.labels)}
        
        return self
    
    def transform(self, X):
        """将图像投影到特征脸空间"""
        return self.pca.transform(X)
    
    def predict(self, X):
        """
        预测图像标签
        
        Args:
            X: 输入图像数据
            
        Returns:
            预测的标签
        """
        # 投影到特征脸空间
        X_transformed = self.transform(X)
        
        # 简单的最近邻分类（可以替换为更复杂的分类器）
        predictions = []
        for i in range(len(X_transformed)):
            # 计算与训练集中每个样本的距离
            distances = []
            for j in range(len(self.training_features_)):
                dist = np.linalg.norm(X_transformed[i] - self.training_features_[j])
                distances.append(dist)
            
            # 找到最近的样本
            min_idx = np.argmin(distances)
            predictions.append(self.training_labels_[min_idx])
        
        return np.array(predictions)
    
    def fit_predict(self, X_train, y_train, X_test):
        """
        训练并预测
        
        Args:
            X_train: 训练数据
            y_train: 训练标签
            X_test: 测试数据
            
        Returns:
            预测结果
        """
        # 训练PCA
        self.fit(X_train, y_train)
        
        # 保存训练数据的特征脸表示
        self.training_features_ = self.transform(X_train)
        self.training_labels_ = y_train
        
        # 预测测试数据
        return self.predict(X_test)


def load_face_data():
    """加载人脸数据集"""
    print("正在加载人脸数据集...")
    
    # 使用LFW数据集
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    
    # 获取数据和标签
    X = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names
    
    print(f"数据集形状: {X.shape}")
    print(f"类别数量: {len(target_names)}")
    print(f"类别名称: {target_names}")
    
    return X, y, target_names


def visualize_eigenfaces(pca, n_eigenfaces=16, image_shape=(50, 37)):
    """
    可视化特征脸
    
    Args:
        pca: 训练好的PCA模型
        n_eigenfaces: 显示的特征脸数量
        image_shape: 图像形状
    """
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('特征脸 (Eigenfaces)', fontsize=16)
    
    for i in range(min(n_eigenfaces, pca.components_.shape[1])):
        row = i // 4
        col = i % 4
        
        # 重构特征脸
        eigenface = pca.components_[:, i].reshape(image_shape)
        
        axes[row, col].imshow(eigenface, cmap='gray')
        axes[row, col].set_title(f'特征脸 {i+1}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('eigenfaces_1d.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_explained_variance(pca):
    """绘制解释方差比"""
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('主成分数量')
    plt.ylabel('累积解释方差比')
    plt.title('1D-PCA 累积解释方差比')
    plt.grid(True)
    plt.savefig('explained_variance_1d.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    print("=== 1D-PCA 特征脸识别实验 ===")
    
    # 加载数据
    X, y, target_names = load_face_data()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 创建并训练特征脸识别器
    recognizer = EigenfaceRecognizer(n_components=100)
    
    print("\n正在训练1D-PCA模型...")
    recognizer.fit(X_train, y_train)
    
    # 可视化特征脸
    print("正在生成特征脸可视化...")
    visualize_eigenfaces(recognizer.pca)
    
    # 绘制解释方差比
    print("正在绘制解释方差比...")
    plot_explained_variance(recognizer.pca)
    
    # 预测测试集
    print("正在预测测试集...")
    y_pred = recognizer.fit_predict(X_train, y_train, X_test)
    
    # 计算准确率
    accuracy = np.mean(y_pred == y_test)
    print(f"\n1D-PCA 特征脸识别准确率: {accuracy:.4f}")
    
    # 生成分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('1D-PCA 特征脸识别混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_1d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n1D-PCA 特征脸识别实验完成！")


if __name__ == "__main__":
    main()
