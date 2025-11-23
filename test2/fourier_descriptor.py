"""
傅里叶描述子（Fourier Descriptors）实现
用于形状特征提取和数字识别
"""

import numpy as np
import cv2
from scipy import fftpack
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class FourierDescriptor:
    """傅里叶描述子类"""
    
    def __init__(self, n_descriptors=20):
        """
        初始化傅里叶描述子
        
        Args:
            n_descriptors: 保留的傅里叶描述子数量
        """
        self.n_descriptors = n_descriptors
        
    def extract_contour(self, image):
        """
        提取图像轮廓
        
        Args:
            image: 输入图像（灰度图或二值图）
            
        Returns:
            轮廓点坐标数组
        """
        # 确保图像是二值图
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 二值化
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if len(contours) == 0:
            return None
        
        # 选择最大的轮廓
        contour = max(contours, key=cv2.contourArea)
        
        # 将轮廓转换为复数形式
        contour = contour.reshape(-1, 2)
        
        return contour
    
    def compute_descriptors(self, contour):
        """
        计算傅里叶描述子
        
        Args:
            contour: 轮廓点坐标数组 (N, 2)
            
        Returns:
            傅里叶描述子
        """
        if contour is None or len(contour) < 3:
            return None
        
        # 将轮廓点转换为复数表示 z = x + iy
        complex_contour = contour[:, 0] + 1j * contour[:, 1]
        
        # 计算离散傅里叶变换（DFT）
        fourier_result = fftpack.fft(complex_contour)
        
        # 归一化：平移不变性（去除DC分量）
        # fourier_result[0] = 0
        
        # 缩放不变性：除以第一个非零系数的模
        if np.abs(fourier_result[1]) > 1e-10:
            fourier_result = fourier_result / np.abs(fourier_result[1])
        
        # 旋转不变性：只取模值
        descriptors = np.abs(fourier_result)
        
        # 保留前n个描述子（低频部分）
        n = min(self.n_descriptors, len(descriptors))
        descriptors = descriptors[:n]
        
        # 归一化到[0, 1]
        if np.max(descriptors) > 0:
            descriptors = descriptors / np.max(descriptors)
        
        return descriptors
    
    def extract_features(self, image):
        """
        从图像提取傅里叶描述子特征
        
        Args:
            image: 输入图像
            
        Returns:
            特征向量
        """
        contour = self.extract_contour(image)
        if contour is None:
            # 返回零向量
            return np.zeros(self.n_descriptors)
        
        descriptors = self.compute_descriptors(contour)
        if descriptors is None:
            return np.zeros(self.n_descriptors)
        
        # 确保特征向量长度一致
        if len(descriptors) < self.n_descriptors:
            # 补零
            descriptors = np.pad(descriptors, (0, self.n_descriptors - len(descriptors)))
        
        return descriptors
    
    def reconstruct_contour(self, descriptors, n_points=200):
        """
        从傅里叶描述子重构轮廓
        
        Args:
            descriptors: 傅里叶描述子
            n_points: 重构的点数
            
        Returns:
            重构的轮廓点
        """
        # 扩展描述子到足够长度
        extended_descriptors = np.zeros(n_points, dtype=complex)
        n = min(len(descriptors), n_points)
        extended_descriptors[:n] = descriptors[:n]
        
        # 逆傅里叶变换
        reconstructed = fftpack.ifft(extended_descriptors)
        
        # 提取实部和虚部作为x, y坐标
        x = np.real(reconstructed)
        y = np.imag(reconstructed)
        
        contour = np.column_stack([x, y])
        
        return contour
    
    def visualize_descriptors(self, image, save_path=None):
        """
        可视化傅里叶描述子
        
        Args:
            image: 输入图像
            save_path: 保存路径
        """
        # 提取轮廓
        contour = self.extract_contour(image)
        if contour is None:
            print("无法提取轮廓")
            return
        
        # 计算描述子
        descriptors = self.compute_descriptors(contour)
        if descriptors is None:
            print("无法计算描述子")
            return
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 原始图像
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('原始图像', fontsize=12)
        axes[0, 0].axis('off')
        
        # 轮廓
        axes[0, 1].plot(contour[:, 0], contour[:, 1], 'b-', linewidth=2)
        axes[0, 1].set_title('提取的轮廓', fontsize=12)
        axes[0, 1].set_aspect('equal')
        axes[0, 1].invert_yaxis()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 傅里叶描述子幅度谱
        axes[1, 0].stem(range(len(descriptors)), descriptors, basefmt=' ')
        axes[1, 0].set_title('傅里叶描述子（幅度谱）', fontsize=12)
        axes[1, 0].set_xlabel('频率索引')
        axes[1, 0].set_ylabel('幅度')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 重构轮廓对比
        reconstructed = self.reconstruct_contour(descriptors)
        axes[1, 1].plot(contour[:, 0], contour[:, 1], 'b-', 
                       linewidth=2, label='原始轮廓', alpha=0.7)
        axes[1, 1].plot(reconstructed[:, 0], reconstructed[:, 1], 'r--', 
                       linewidth=2, label='重构轮廓', alpha=0.7)
        axes[1, 1].set_title('轮廓重构对比', fontsize=12)
        axes[1, 1].set_aspect('equal')
        axes[1, 1].invert_yaxis()
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存到: {save_path}")
        
        plt.show()


def compute_distance(desc1, desc2, metric='euclidean'):
    """
    计算两个描述子之间的距离
    
    Args:
        desc1: 描述子1
        desc2: 描述子2
        metric: 距离度量方式 ('euclidean', 'cosine', 'correlation')
        
    Returns:
        距离值
    """
    if metric == 'euclidean':
        return np.linalg.norm(desc1 - desc2)
    elif metric == 'cosine':
        dot_product = np.dot(desc1, desc2)
        norm1 = np.linalg.norm(desc1)
        norm2 = np.linalg.norm(desc2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        return 1 - dot_product / (norm1 * norm2)
    elif metric == 'correlation':
        if np.std(desc1) == 0 or np.std(desc2) == 0:
            return 1.0
        return 1 - np.corrcoef(desc1, desc2)[0, 1]
    else:
        raise ValueError(f"Unknown metric: {metric}")


def test_fourier_descriptor():
    """测试傅里叶描述子"""
    print("=== 傅里叶描述子测试 ===\n")
    
    # 创建一个简单的测试图像（圆形）
    print("创建测试图像...")
    image = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(image, (100, 100), 50, 255, -1)
    
    # 创建傅里叶描述子对象
    fd = FourierDescriptor(n_descriptors=30)
    
    # 提取特征
    print("提取傅里叶描述子...")
    features = fd.extract_features(image)
    print(f"特征维度: {len(features)}")
    print(f"前10个描述子: {features[:10]}")
    
    # 可视化
    print("\n生成可视化...")
    fd.visualize_descriptors(image, save_path='fourier_descriptor_test.png')
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_fourier_descriptor()
