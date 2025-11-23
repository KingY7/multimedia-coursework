"""
基于傅里叶描述子的数字识别系统
支持MNIST手写数字识别
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from fourier_descriptor import FourierDescriptor, compute_distance
import cv2

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class DigitRecognizer:
    """基于傅里叶描述子的数字识别器"""
    
    def __init__(self, n_descriptors=30, classifier_type='knn'):
        """
        初始化识别器
        
        Args:
            n_descriptors: 傅里叶描述子数量
            classifier_type: 分类器类型 ('knn', 'svm')
        """
        self.fd = FourierDescriptor(n_descriptors=n_descriptors)
        self.classifier_type = classifier_type
        
        if classifier_type == 'knn':
            self.classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        elif classifier_type == 'svm':
            self.classifier = SVC(kernel='rbf', C=10, gamma='scale')
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        self.is_trained = False
        
    def extract_features_batch(self, images):
        """
        批量提取特征
        
        Args:
            images: 图像数组 (n_samples, height, width)
            
        Returns:
            特征矩阵 (n_samples, n_descriptors)
        """
        n_samples = len(images)
        features = np.zeros((n_samples, self.fd.n_descriptors))
        
        for i in range(n_samples):
            features[i] = self.fd.extract_features(images[i])
            
            if (i + 1) % 100 == 0:
                print(f"  已处理 {i + 1}/{n_samples} 张图像")
        
        return features
    
    def train(self, X_train, y_train):
        """
        训练识别器
        
        Args:
            X_train: 训练图像
            y_train: 训练标签
        """
        print(f"正在提取训练集特征（共{len(X_train)}张图像）...")
        X_features = self.extract_features_batch(X_train)
        
        print(f"\n正在训练{self.classifier_type.upper()}分类器...")
        self.classifier.fit(X_features, y_train)
        
        self.is_trained = True
        print("训练完成！")
        
    def predict(self, X_test):
        """
        预测数字
        
        Args:
            X_test: 测试图像
            
        Returns:
            预测标签
        """
        if not self.is_trained:
            raise RuntimeError("模型未训练，请先调用train()方法")
        
        print(f"正在提取测试集特征（共{len(X_test)}张图像）...")
        X_features = self.extract_features_batch(X_test)
        
        print("正在预测...")
        predictions = self.classifier.predict(X_features)
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        评估模型性能
        
        Args:
            X_test: 测试图像
            y_test: 测试标签
            
        Returns:
            准确率
        """
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, y_pred


def load_mnist_data():
    """加载MNIST数字数据集（sklearn内置版本）"""
    print("正在加载数字数据集...")
    
    # 使用sklearn内置的8x8数字数据集
    digits = load_digits()
    
    # 获取图像和标签
    images = digits.images  # (n_samples, 8, 8)
    labels = digits.target
    
    # 放大图像以便更好地提取轮廓
    enlarged_images = []
    for img in images:
        # 归一化到0-255
        img_normalized = (img / 16.0 * 255).astype(np.uint8)
        # 放大到28x28
        enlarged = cv2.resize(img_normalized, (28, 28), interpolation=cv2.INTER_LINEAR)
        enlarged_images.append(enlarged)
    
    enlarged_images = np.array(enlarged_images)
    
    print(f"数据集大小: {enlarged_images.shape}")
    print(f"类别数量: {len(np.unique(labels))}")
    print(f"类别: {np.unique(labels)}")
    
    return enlarged_images, labels


def visualize_samples(images, labels, n_samples=10):
    """可视化样本"""
    print("\n正在生成样本展示...")
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('数字样本展示', fontsize=16, fontweight='bold')
    
    for i in range(n_samples):
        row = i // 5
        col = i % 5
        
        axes[row, col].imshow(images[i], cmap='gray')
        axes[row, col].set_title(f'数字: {labels[i]}', fontsize=12)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('digit_samples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("样本展示已保存: digit_samples.png")


def visualize_fourier_features(recognizer, images, labels):
    """可视化不同数字的傅里叶描述子"""
    print("\n正在生成傅里叶描述子可视化...")
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('不同数字的傅里叶描述子特征', fontsize=16, fontweight='bold')
    
    for digit in range(10):
        # 找到该数字的第一个样本
        idx = np.where(labels == digit)[0][0]
        image = images[idx]
        
        # 提取特征
        features = recognizer.fd.extract_features(image)
        
        row = digit // 5
        col = digit % 5
        
        axes[row, col].stem(range(len(features)), features, basefmt=' ')
        axes[row, col].set_title(f'数字 {digit}', fontsize=11)
        axes[row, col].set_xlabel('频率索引', fontsize=9)
        axes[row, col].set_ylabel('幅度', fontsize=9)
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('fourier_features_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("特征对比已保存: fourier_features_comparison.png")


def plot_confusion_matrix(y_test, y_pred):
    """绘制混淆矩阵"""
    print("\n正在生成混淆矩阵...")
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10),
                cbar_kws={'label': '数量'})
    plt.title('数字识别混淆矩阵', fontsize=16, fontweight='bold')
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("混淆矩阵已保存: confusion_matrix.png")


def plot_recognition_examples(recognizer, X_test, y_test, y_pred, n_examples=10):
    """展示识别示例"""
    print("\n正在生成识别示例...")
    
    # 找出正确和错误的预测
    correct_idx = np.where(y_pred == y_test)[0]
    wrong_idx = np.where(y_pred != y_test)[0]
    
    n_correct = min(5, len(correct_idx))
    n_wrong = min(5, len(wrong_idx))
    
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle('识别示例（上：正确，下：错误）', fontsize=16, fontweight='bold')
    
    # 正确的示例
    for i in range(n_correct):
        idx = correct_idx[i]
        axes[0, i].imshow(X_test[idx], cmap='gray')
        axes[0, i].set_title(f'真实:{y_test[idx]}\n预测:{y_pred[idx]}', 
                            fontsize=10, color='green')
        axes[0, i].axis('off')
    
    # 填充空白
    for i in range(n_correct, 5):
        axes[0, i].axis('off')
    
    # 错误的示例
    for i in range(n_wrong):
        idx = wrong_idx[i]
        axes[1, i].imshow(X_test[idx], cmap='gray')
        axes[1, i].set_title(f'真实:{y_test[idx]}\n预测:{y_pred[idx]}', 
                            fontsize=10, color='red')
        axes[1, i].axis('off')
    
    # 填充空白
    for i in range(n_wrong, 5):
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('recognition_examples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("识别示例已保存: recognition_examples.png")


def main():
    """主函数"""
    print("=" * 70)
    print("基于傅里叶描述子的数字识别实验")
    print("=" * 70)
    
    # 加载数据
    images, labels = load_mnist_data()
    
    # 可视化样本
    visualize_samples(images, labels)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"\n训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 创建识别器（使用KNN）
    print("\n" + "=" * 70)
    print("使用K近邻分类器")
    print("=" * 70)
    
    recognizer_knn = DigitRecognizer(n_descriptors=30, classifier_type='knn')
    
    # 训练模型
    recognizer_knn.train(X_train, y_train)
    
    # 可视化傅里叶特征
    visualize_fourier_features(recognizer_knn, images, labels)
    
    # 评估模型
    accuracy_knn, y_pred_knn = recognizer_knn.evaluate(X_test, y_test)
    
    print(f"\n{'=' * 70}")
    print(f"K近邻分类器准确率: {accuracy_knn:.4f} ({accuracy_knn*100:.2f}%)")
    print("=" * 70)
    
    # 详细分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred_knn, 
                                target_names=[str(i) for i in range(10)]))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(y_test, y_pred_knn)
    
    # 展示识别示例
    plot_recognition_examples(recognizer_knn, X_test, y_test, y_pred_knn)
    
    # 使用SVM分类器
    print("\n" + "=" * 70)
    print("使用支持向量机分类器")
    print("=" * 70)
    
    recognizer_svm = DigitRecognizer(n_descriptors=30, classifier_type='svm')
    recognizer_svm.train(X_train, y_train)
    
    accuracy_svm, y_pred_svm = recognizer_svm.evaluate(X_test, y_test)
    
    print(f"\n{'=' * 70}")
    print(f"支持向量机准确率: {accuracy_svm:.4f} ({accuracy_svm*100:.2f}%)")
    print("=" * 70)
    
    # 对比结果
    print("\n" + "=" * 70)
    print("分类器性能对比")
    print("=" * 70)
    print(f"K近邻 (KNN):      {accuracy_knn:.4f} ({accuracy_knn*100:.2f}%)")
    print(f"支持向量机 (SVM): {accuracy_svm:.4f} ({accuracy_svm*100:.2f}%)")
    
    if accuracy_knn > accuracy_svm:
        print("\n最佳分类器: K近邻 (KNN)")
    else:
        print("\n最佳分类器: 支持向量机 (SVM)")
    
    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)
    
    print("\n生成的文件:")
    print("  - digit_samples.png (数字样本)")
    print("  - fourier_features_comparison.png (傅里叶特征对比)")
    print("  - confusion_matrix.png (混淆矩阵)")
    print("  - recognition_examples.png (识别示例)")


if __name__ == "__main__":
    main()
