"""
PCA特征脸可视化分析脚本
提供更详细的可视化分析和性能评估
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA as SklearnPCA
import seaborn as sns
from pca_1d import PCA1D, EigenfaceRecognizer
from pca_2d import PCA2D, Eigenface2DRecognizer

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """加载人脸数据集"""
    print("正在加载人脸数据集...")
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    
    X = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names
    image_shape = (50, 37)
    
    return X, y, target_names, image_shape


def plot_component_analysis(X_train, y_train, image_shape):
    """分析不同主成分数量对性能的影响"""
    print("\n正在分析主成分数量的影响...")
    
    component_range = [10, 20, 30, 50, 75, 100, 150, 200]
    accuracies_1d = []
    accuracies_2d = []
    training_times_1d = []
    training_times_2d = []
    
    # 简单验证集
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    for n_comp in component_range:
        print(f"  测试 {n_comp} 个主成分...")
        
        # 1D-PCA
        import time
        start = time.time()
        recognizer_1d = EigenfaceRecognizer(n_components=n_comp)
        recognizer_1d.fit(X_tr, y_tr)
        recognizer_1d.training_features_ = recognizer_1d.transform(X_tr)
        recognizer_1d.training_labels_ = y_tr
        training_times_1d.append(time.time() - start)
        
        y_pred_1d = recognizer_1d.predict(X_val)
        acc_1d = np.mean(y_pred_1d == y_val)
        accuracies_1d.append(acc_1d)
        
        # 2D-PCA (调整成分数量以适应图像宽度)
        n_comp_2d = min(n_comp, image_shape[1])
        start = time.time()
        recognizer_2d = Eigenface2DRecognizer(n_components=n_comp_2d, image_shape=image_shape)
        recognizer_2d.fit(X_tr, y_tr)
        recognizer_2d.training_features_ = recognizer_2d.transform(X_tr)
        recognizer_2d.training_labels_ = y_tr
        training_times_2d.append(time.time() - start)
        
        y_pred_2d = recognizer_2d.predict(X_val)
        acc_2d = np.mean(y_pred_2d == y_val)
        accuracies_2d.append(acc_2d)
    
    # 绘制结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 准确率曲线
    ax1.plot(component_range, accuracies_1d, 'o-', label='1D-PCA', linewidth=2, markersize=8)
    ax1.plot(component_range, accuracies_2d, 's-', label='2D-PCA', linewidth=2, markersize=8)
    ax1.set_xlabel('主成分数量', fontsize=12)
    ax1.set_ylabel('验证集准确率', fontsize=12)
    ax1.set_title('主成分数量对识别准确率的影响', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 训练时间曲线
    ax2.plot(component_range, training_times_1d, 'o-', label='1D-PCA', linewidth=2, markersize=8)
    ax2.plot(component_range, training_times_2d, 's-', label='2D-PCA', linewidth=2, markersize=8)
    ax2.set_xlabel('主成分数量', fontsize=12)
    ax2.set_ylabel('训练时间 (秒)', fontsize=12)
    ax2.set_title('主成分数量对训练时间的影响', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('component_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_feature_space(X_train, y_train, target_names, image_shape):
    """使用t-SNE可视化特征空间"""
    print("\n正在生成特征空间可视化...")
    
    # 限制样本数量以加快t-SNE计算
    max_samples = 300
    if len(X_train) > max_samples:
        indices = np.random.choice(len(X_train), max_samples, replace=False)
        X_subset = X_train[indices]
        y_subset = y_train[indices]
    else:
        X_subset = X_train
        y_subset = y_train
    
    # 1D-PCA特征
    recognizer_1d = EigenfaceRecognizer(n_components=50)
    recognizer_1d.fit(X_subset, y_subset)
    features_1d = recognizer_1d.transform(X_subset)
    
    # 2D-PCA特征
    recognizer_2d = Eigenface2DRecognizer(n_components=30, image_shape=image_shape)
    recognizer_2d.fit(X_subset, y_subset)
    features_2d = recognizer_2d.transform(X_subset)
    
    # t-SNE降维到2D
    print("  应用t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_1d_2d = tsne.fit_transform(features_1d)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d_2d = tsne.fit_transform(features_2d)
    
    # 绘制结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1D-PCA特征空间
    scatter1 = ax1.scatter(features_1d_2d[:, 0], features_1d_2d[:, 1], 
                           c=y_subset, cmap='tab10', s=50, alpha=0.7)
    ax1.set_title('1D-PCA特征空间 (t-SNE可视化)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE维度 1', fontsize=12)
    ax1.set_ylabel('t-SNE维度 2', fontsize=12)
    
    # 添加图例
    handles, labels = scatter1.legend_elements()
    ax1.legend(handles, target_names, title="人物", loc='best', fontsize=9)
    
    # 2D-PCA特征空间
    scatter2 = ax2.scatter(features_2d_2d[:, 0], features_2d_2d[:, 1], 
                           c=y_subset, cmap='tab10', s=50, alpha=0.7)
    ax2.set_title('2D-PCA特征空间 (t-SNE可视化)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE维度 1', fontsize=12)
    ax2.set_ylabel('t-SNE维度 2', fontsize=12)
    
    # 添加图例
    handles, labels = scatter2.legend_elements()
    ax2.legend(handles, target_names, title="人物", loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('feature_space_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_sample_faces(X, y, target_names, n_samples=5):
    """可视化每个人物的样本人脸"""
    print("\n正在生成样本人脸展示...")
    
    n_classes = len(target_names)
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(12, 14))
    fig.suptitle('每个人物的样本人脸', fontsize=16, fontweight='bold')
    
    for i, name in enumerate(target_names):
        # 获取该人物的所有样本
        class_samples = X[y == i]
        
        # 随机选择n_samples个样本
        if len(class_samples) >= n_samples:
            indices = np.random.choice(len(class_samples), n_samples, replace=False)
        else:
            indices = np.arange(len(class_samples))
        
        for j, idx in enumerate(indices):
            if j < n_samples:
                img = class_samples[idx].reshape(50, 37)
                axes[i, j].imshow(img, cmap='gray')
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_title(name, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('sample_faces.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_reconstruction_quality(X_test, image_shape):
    """比较不同主成分数量下的重构质量"""
    print("\n正在分析重构质量...")
    
    # 选择一个测试样本
    sample = X_test[0:1]
    
    component_counts = [10, 30, 50, 100, 150]
    
    fig, axes = plt.subplots(2, len(component_counts) + 1, figsize=(16, 6))
    fig.suptitle('不同主成分数量的重构质量对比', fontsize=16, fontweight='bold')
    
    # 显示原始图像
    axes[0, 0].imshow(sample.reshape(image_shape), cmap='gray')
    axes[0, 0].set_title('原始图像', fontsize=11)
    axes[0, 0].axis('off')
    axes[1, 0].imshow(sample.reshape(image_shape), cmap='gray')
    axes[1, 0].set_title('原始图像', fontsize=11)
    axes[1, 0].axis('off')
    
    # 不同主成分数量的重构
    for idx, n_comp in enumerate(component_counts):
        # 1D-PCA重构
        pca_1d = PCA1D(n_components=n_comp)
        pca_1d.fit(X_test[:100])  # 用部分数据训练
        features = pca_1d.transform(sample)
        reconstructed = pca_1d.inverse_transform(features)
        
        axes[0, idx + 1].imshow(reconstructed.reshape(image_shape), cmap='gray')
        axes[0, idx + 1].set_title(f'1D-PCA\n{n_comp}成分', fontsize=10)
        axes[0, idx + 1].axis('off')
        
        # 2D-PCA重构
        n_comp_2d = min(n_comp, image_shape[1])
        pca_2d = PCA2D(n_components=n_comp_2d)
        pca_2d.fit(X_test[:100], image_shape)
        features = pca_2d.transform(sample)
        reconstructed = pca_2d.inverse_transform(features)
        
        axes[1, idx + 1].imshow(reconstructed.reshape(image_shape), cmap='gray')
        axes[1, idx + 1].set_title(f'2D-PCA\n{n_comp_2d}成分', fontsize=10)
        axes[1, idx + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('reconstruction_quality.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_eigenface_details(X_train, y_train, image_shape):
    """详细展示特征脸的形成过程"""
    print("\n正在生成特征脸形成过程可视化...")
    
    # 训练PCA
    pca_1d = PCA1D(n_components=100)
    pca_1d.fit(X_train)
    
    # 创建图表
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 显示平均脸
    ax1 = plt.subplot(2, 4, 1)
    mean_face = pca_1d.mean_.reshape(image_shape)
    ax1.imshow(mean_face, cmap='gray')
    ax1.set_title('平均脸', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. 显示前7个特征脸
    for i in range(7):
        ax = plt.subplot(2, 4, i + 2)
        eigenface = pca_1d.components_[:, i].reshape(image_shape)
        ax.imshow(eigenface, cmap='gray')
        variance_pct = pca_1d.explained_variance_ratio_[i] * 100
        ax.set_title(f'特征脸{i+1}\n方差占比:{variance_pct:.2f}%', fontsize=10)
        ax.axis('off')
    
    plt.suptitle('1D-PCA 特征脸详细分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eigenface_details.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_variance_explained_detailed(X_train, y_train, image_shape):
    """详细的方差解释率分析"""
    print("\n正在生成方差解释率详细分析...")
    
    # 训练模型
    pca_1d = PCA1D(n_components=200)
    pca_1d.fit(X_train)
    
    pca_2d = PCA2D(n_components=37)  # 图像宽度
    pca_2d.fit(X_train, image_shape)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('方差解释率详细分析', fontsize=16, fontweight='bold')
    
    # 1D-PCA单个成分的方差贡献
    ax1 = axes[0, 0]
    ax1.bar(range(50), pca_1d.explained_variance_ratio_[:50], alpha=0.7, color='steelblue')
    ax1.set_xlabel('主成分索引')
    ax1.set_ylabel('方差解释率')
    ax1.set_title('1D-PCA 前50个主成分方差贡献')
    ax1.grid(axis='y', alpha=0.3)
    
    # 1D-PCA累积方差
    ax2 = axes[0, 1]
    cumsum_1d = np.cumsum(pca_1d.explained_variance_ratio_)
    ax2.plot(cumsum_1d, linewidth=2, color='steelblue')
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95%阈值')
    ax2.axhline(y=0.90, color='orange', linestyle='--', label='90%阈值')
    ax2.set_xlabel('主成分数量')
    ax2.set_ylabel('累积方差解释率')
    ax2.set_title('1D-PCA 累积方差解释率')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 找到达到95%的主成分数量
    n_comp_95 = np.argmax(cumsum_1d >= 0.95) + 1
    ax2.scatter([n_comp_95], [0.95], color='r', s=100, zorder=5)
    ax2.text(n_comp_95, 0.95, f'  {n_comp_95}个成分', fontsize=9)
    
    # 2D-PCA单个成分的方差贡献
    ax3 = axes[1, 0]
    ax3.bar(range(len(pca_2d.explained_variance_ratio_)), 
            pca_2d.explained_variance_ratio_, alpha=0.7, color='seagreen')
    ax3.set_xlabel('主成分索引')
    ax3.set_ylabel('方差解释率')
    ax3.set_title('2D-PCA 主成分方差贡献')
    ax3.grid(axis='y', alpha=0.3)
    
    # 2D-PCA累积方差
    ax4 = axes[1, 1]
    cumsum_2d = np.cumsum(pca_2d.explained_variance_ratio_)
    ax4.plot(cumsum_2d, linewidth=2, color='seagreen')
    ax4.axhline(y=0.95, color='r', linestyle='--', label='95%阈值')
    ax4.axhline(y=0.90, color='orange', linestyle='--', label='90%阈值')
    ax4.set_xlabel('主成分数量')
    ax4.set_ylabel('累积方差解释率')
    ax4.set_title('2D-PCA 累积方差解释率')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 找到达到95%的主成分数量
    if np.any(cumsum_2d >= 0.95):
        n_comp_95_2d = np.argmax(cumsum_2d >= 0.95) + 1
        ax4.scatter([n_comp_95_2d], [0.95], color='r', s=100, zorder=5)
        ax4.text(n_comp_95_2d, 0.95, f'  {n_comp_95_2d}个成分', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('variance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    print("=" * 60)
    print("PCA特征脸可视化分析")
    print("=" * 60)
    
    # 加载数据
    X, y, target_names, image_shape = load_data()
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\n数据集信息:")
    print(f"  训练集: {X_train.shape}")
    print(f"  测试集: {X_test.shape}")
    print(f"  类别数: {len(target_names)}")
    
    # 1. 样本人脸展示
    visualize_sample_faces(X, y, target_names)
    
    # 2. 主成分数量分析
    plot_component_analysis(X_train, y_train, image_shape)
    
    # 3. 特征空间可视化
    visualize_feature_space(X_train, y_train, target_names, image_shape)
    
    # 4. 特征脸详细分析
    visualize_eigenface_details(X_train, y_train, image_shape)
    
    # 5. 方差解释率分析
    plot_variance_explained_detailed(X_train, y_train, image_shape)
    
    # 6. 重构质量分析
    compare_reconstruction_quality(X_test, image_shape)
    
    print("\n" + "=" * 60)
    print("可视化分析完成！")
    print("生成的文件:")
    print("  - sample_faces.png (样本人脸展示)")
    print("  - component_analysis.png (主成分数量分析)")
    print("  - feature_space_visualization.png (特征空间可视化)")
    print("  - eigenface_details.png (特征脸详细分析)")
    print("  - variance_analysis.png (方差解释率分析)")
    print("  - reconstruction_quality.png (重构质量对比)")
    print("=" * 60)


if __name__ == "__main__":
    main()
