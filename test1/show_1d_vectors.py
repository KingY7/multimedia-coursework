"""
1D-PCA 特征向量和投影向量展示（修复中文显示）
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from pca_1d import PCA1D

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    return lfw_people.data, lfw_people.target, lfw_people.target_names, (50, 37)

# 加载数据
X, y, names, shape = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 训练1D-PCA
pca = PCA1D(n_components=100)
pca.fit(X_train)

# ========== 图1: 1D-PCA特征向量 ==========
fig, axes = plt.subplots(4, 5, figsize=(16, 12))
fig.suptitle('1D-PCA 特征向量（特征脸）', fontsize=18, fontweight='bold')

# 显示平均脸
axes[0, 0].imshow(pca.mean_.reshape(shape), cmap='gray')
axes[0, 0].set_title('平均脸', fontsize=12, color='red', weight='bold')
axes[0, 0].axis('off')

# 显示说明
axes[0, 1].axis('off')
axes[0, 1].text(0.1, 0.5, 
    f'特征向量矩阵形状:\n{pca.components_.shape}\n\n'
    f'每个特征向量:\n{pca.components_.shape[0]}维\n\n'
    f'含义: 人脸变化的主要方向',
    fontsize=11, va='center',
    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

# 显示前18个特征向量
for i in range(18):
    row = (i + 2) // 5
    col = (i + 2) % 5
    eigenface = pca.components_[:, i].reshape(shape)
    axes[row, col].imshow(eigenface, cmap='RdBu_r')
    var = pca.explained_variance_ratio_[i] * 100
    axes[row, col].set_title(f'#{i+1} 方差{var:.1f}%', fontsize=9)
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('1D特征向量展示.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 1D特征向量展示.png")
plt.show()

# ========== 图2: 1D-PCA投影向量 ==========
fig = plt.figure(figsize=(18, 10))
fig.suptitle('1D-PCA 投影向量（降维后的特征表示）', fontsize=18, fontweight='bold')

samples = [0, 50, 100]
for idx, s in enumerate(samples):
    # 原图
    ax1 = plt.subplot(3, 6, idx*6 + 1)
    ax1.imshow(X_train[s].reshape(shape), cmap='gray')
    ax1.set_title(f'{names[y_train[s]]}', fontsize=10, weight='bold')
    ax1.axis('off')
    
    # 投影向量
    proj = pca.transform(X_train[s:s+1])[0]
    ax2 = plt.subplot(3, 6, (idx*6 + 2, idx*6 + 3))
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    bars = ax2.bar(range(10), proj[:10], color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('特征向量索引', fontsize=10)
    ax2.set_ylabel('投影系数', fontsize=10)
    ax2.set_title(f'投影向量 前10个系数 共{len(proj)}维', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(0, color='black', linewidth=1)
    for bar, val in zip(bars, proj[:10]):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h, f'{val:.0f}',
                ha='center', va='bottom' if val>0 else 'top', fontsize=8)
    
    # 重构图像
    for j, n in enumerate([5, 20, 100]):
        ax3 = plt.subplot(3, 6, idx*6 + 4 + j)
        recon = np.dot(proj[:n], pca.components_[:, :n].T) + pca.mean_
        ax3.imshow(recon.reshape(shape), cmap='gray')
        ax3.set_title(f'重构 {n}成分', fontsize=9)
        ax3.axis('off')

plt.tight_layout()
plt.savefig('1D投影向量展示.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 1D投影向量展示.png")
plt.show()

# ========== 图3: 投影过程详解 ==========
fig = plt.figure(figsize=(18, 7))
fig.suptitle('1D-PCA 投影过程详细步骤', fontsize=18, fontweight='bold')

s = 0
original = X_train[s]
proj = pca.transform(original.reshape(1, -1))[0]

# 步骤1: 原图
ax1 = plt.subplot(2, 6, 1)
ax1.imshow(original.reshape(shape), cmap='gray')
ax1.set_title(f'步骤1\n原始图像\n{names[y_train[s]]}\n维度{original.shape[0]}', fontsize=10, weight='bold')
ax1.axis('off')

# 步骤2: 平均脸
ax2 = plt.subplot(2, 6, 2)
ax2.imshow(pca.mean_.reshape(shape), cmap='gray')
ax2.set_title('步骤2\n平均脸', fontsize=10, weight='bold')
ax2.axis('off')

# 步骤3: 中心化
ax3 = plt.subplot(2, 6, 3)
centered = (original - pca.mean_).reshape(shape)
ax3.imshow(centered, cmap='RdBu_r')
ax3.set_title('步骤3\n中心化\n原图减平均', fontsize=10, weight='bold')
ax3.axis('off')

# 步骤4-5: 特征向量
for i in range(3):
    ax = plt.subplot(2, 6, 4+i)
    eigenface = pca.components_[:, i].reshape(shape)
    ax.imshow(eigenface, cmap='RdBu_r')
    ax.set_title(f'步骤4\n特征向量{i+1}', fontsize=10, weight='bold')
    ax.axis('off')

# 计算说明
ax_calc = plt.subplot(2, 6, 7)
ax_calc.axis('off')
ax_calc.text(0.1, 0.5,
    f'步骤5 投影计算\n\n'
    f'投影系数 = 中心化图像 点乘 特征向量\n\n'
    f'系数1 = {proj[0]:.1f}\n'
    f'系数2 = {proj[1]:.1f}\n'
    f'系数3 = {proj[2]:.1f}\n'
    f'...\n'
    f'共{len(proj)}个系数',
    fontsize=10, va='center',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 投影向量可视化
ax_proj = plt.subplot(2, 6, (8, 12))
n = 30
colors = plt.cm.viridis(np.linspace(0, 1, n))
ax_proj.barh(range(n), proj[:n], color=colors, edgecolor='black')
ax_proj.set_xlabel('投影系数值', fontsize=11)
ax_proj.set_ylabel('特征向量索引', fontsize=11)
ax_proj.set_title(f'步骤6 投影向量\n前{n}个系数', fontsize=12, weight='bold')
ax_proj.invert_yaxis()
ax_proj.grid(axis='x', alpha=0.3)
ax_proj.axvline(0, color='black', linewidth=1)

plt.tight_layout()
plt.savefig('1D投影过程详解.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 1D投影过程详解.png")
plt.show()

print("\n1D-PCA 可视化完成!")
