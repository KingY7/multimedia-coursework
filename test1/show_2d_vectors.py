"""
2D-PCA 特征向量和投影向量展示（修复中文显示）
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from pca_2d import PCA2D

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    return lfw_people.data, lfw_people.target, lfw_people.target_names, (50, 37)

# 加载数据
X, y, names, shape = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 训练2D-PCA
pca = PCA2D(n_components=30)
pca.fit(X_train, shape)

# ========== 图1: 2D-PCA特征向量 ==========
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.suptitle('2D-PCA 特征向量（列向量）', fontsize=18, fontweight='bold')

# 显示平均图像
axes[0, 0].imshow(pca.mean_, cmap='gray')
axes[0, 0].set_title('平均脸矩阵', fontsize=12, color='red', weight='bold')
axes[0, 0].axis('off')

# 显示说明
axes[0, 1].axis('off')
axes[0, 1].text(0.1, 0.5,
    f'特征向量矩阵形状:\n{pca.components_.shape}\n\n'
    f'每个特征向量:\n{pca.components_.shape[0]}维\n'
    f'(图像宽度)\n\n'
    f'含义: 列方向的主要模式',
    fontsize=11, va='center',
    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# 显示前10个特征向量
for i in range(10):
    row = (i + 2) // 4
    col = (i + 2) % 4
    if row < 3:
        eigenvec = pca.components_[:, i]
        axes[row, col].plot(eigenvec, linewidth=3, color=f'C{i}', marker='o', markersize=3)
        axes[row, col].fill_between(range(len(eigenvec)), eigenvec, alpha=0.3, color=f'C{i}')
        axes[row, col].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axes[row, col].grid(True, alpha=0.3)
        var = pca.explained_variance_ratio_[i] * 100
        axes[row, col].set_title(f'特征向量{i+1} 方差{var:.1f}%', fontsize=10)
        axes[row, col].set_xlabel('列索引', fontsize=9)
        axes[row, col].set_ylabel('值', fontsize=9)

plt.tight_layout()
plt.savefig('2D特征向量展示.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 2D特征向量展示.png")
plt.show()

# ========== 图2: 2D-PCA投影向量 ==========
fig = plt.figure(figsize=(18, 10))
fig.suptitle('2D-PCA 投影向量（降维后的特征表示）', fontsize=18, fontweight='bold')

samples = [0, 50, 100]
for idx, s in enumerate(samples):
    # 原图
    ax1 = plt.subplot(3, 6, idx*6 + 1)
    ax1.imshow(X_train[s].reshape(shape), cmap='gray')
    ax1.set_title(f'{names[y_train[s]]}', fontsize=10, weight='bold')
    ax1.axis('off')
    
    # 投影向量（展平后）
    proj = pca.transform(X_train[s:s+1])[0]  # shape: (50*30,)
    proj_matrix = proj.reshape(shape[0], 30)  # 重塑为矩阵形式 (50, 30)
    
    # 显示投影矩阵
    ax2 = plt.subplot(3, 6, (idx*6 + 2, idx*6 + 3))
    im = ax2.imshow(proj_matrix, cmap='RdYlBu', aspect='auto')
    ax2.set_xlabel('主成分索引', fontsize=10)
    ax2.set_ylabel('行索引', fontsize=10)
    ax2.set_title(f'投影矩阵 {proj_matrix.shape}\n共{len(proj)}维', fontsize=10)
    plt.colorbar(im, ax=ax2)
    
    # 重构图像
    for j, n in enumerate([5, 15, 30]):
        ax3 = plt.subplot(3, 6, idx*6 + 4 + j)
        # 使用前n个主成分重构
        proj_partial = proj_matrix[:, :n]
        comp_partial = pca.components_[:, :n]
        recon_matrix = np.dot(proj_partial, comp_partial.T) + pca.mean_
        ax3.imshow(recon_matrix, cmap='gray')
        ax3.set_title(f'重构 {n}成分', fontsize=9)
        ax3.axis('off')

plt.tight_layout()
plt.savefig('2D投影向量展示.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 2D投影向量展示.png")
plt.show()

# ========== 图3: 2D投影过程详解 ==========
fig = plt.figure(figsize=(18, 8))
fig.suptitle('2D-PCA 投影过程详细步骤', fontsize=18, fontweight='bold')

s = 0
original = X_train[s].reshape(shape)
proj = pca.transform(X_train[s:s+1])[0]
proj_matrix = proj.reshape(shape[0], 30)

# 步骤1: 原图矩阵
ax1 = plt.subplot(2, 5, 1)
ax1.imshow(original, cmap='gray')
ax1.set_title(f'步骤1\n原始图像矩阵\n{names[y_train[s]]}\n形状{shape}', fontsize=10, weight='bold')
ax1.axis('off')

# 步骤2: 平均矩阵
ax2 = plt.subplot(2, 5, 2)
ax2.imshow(pca.mean_, cmap='gray')
ax2.set_title(f'步骤2\n平均矩阵\n形状{pca.mean_.shape}', fontsize=10, weight='bold')
ax2.axis('off')

# 步骤3: 中心化矩阵
ax3 = plt.subplot(2, 5, 3)
centered = original - pca.mean_
ax3.imshow(centered, cmap='RdBu_r')
ax3.set_title(f'步骤3\n中心化矩阵\n原图减平均', fontsize=10, weight='bold')
ax3.axis('off')

# 步骤4: 特征向量
ax4 = plt.subplot(2, 5, 4)
eigenvec1 = pca.components_[:, 0]
ax4.plot(eigenvec1, linewidth=4, color='darkgreen', marker='o', markersize=4)
ax4.fill_between(range(len(eigenvec1)), eigenvec1, alpha=0.3, color='green')
ax4.axhline(0, color='black', linewidth=1, linestyle='--')
ax4.grid(True, alpha=0.3)
ax4.set_title('步骤4\n第1个特征向量\n37维列向量', fontsize=10, weight='bold')
ax4.set_xlabel('维度', fontsize=9)

# 计算说明
ax_calc = plt.subplot(2, 5, 5)
ax_calc.axis('off')
ax_calc.text(0.1, 0.5,
    f'步骤5 投影计算\n\n'
    f'投影矩阵 = 中心化矩阵 × 特征向量矩阵\n\n'
    f'Y = A × X\n'
    f'{shape} × {pca.components_[:,:5].shape}\n'
    f'= {proj_matrix[:,:5].shape}\n\n'
    f'展平后维度: {len(proj)}',
    fontsize=10, va='center',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 投影矩阵可视化
ax_proj = plt.subplot(2, 5, (6, 10))
im = ax_proj.imshow(proj_matrix, cmap='RdYlBu', aspect='auto')
ax_proj.set_xlabel('主成分索引 (30个)', fontsize=11)
ax_proj.set_ylabel('行索引 (50行)', fontsize=11)
ax_proj.set_title(f'步骤6 投影矩阵\n形状 {proj_matrix.shape}', fontsize=12, weight='bold')
plt.colorbar(im, ax=ax_proj, label='投影系数值')

plt.tight_layout()
plt.savefig('2D投影过程详解.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 2D投影过程详解.png")
plt.show()

print("\n2D-PCA 可视化完成!")
