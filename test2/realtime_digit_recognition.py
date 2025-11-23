"""
基于傅里叶描述子的实时数字识别系统
使用摄像头捕获手写数字并进行实时识别
"""

import cv2
import numpy as np
import pickle
import os
from fourier_descriptor import FourierDescriptor
from digit_recognition import DigitRecognizer, load_mnist_data
from sklearn.model_selection import train_test_split

# 设置中文字体（用于保存图片时的标注）
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


class RealtimeDigitRecognizer:
    """实时数字识别器"""
    
    def __init__(self, model_path='digit_model.pkl'):
        """
        初始化实时识别器
        
        Args:
            model_path: 训练好的模型路径
        """
        self.model_path = model_path
        self.recognizer = None
        self.fd = FourierDescriptor(n_descriptors=30)
        
        # 图像处理参数
        self.roi_size = 200  # ROI区域大小
        self.threshold_value = 127  # 二值化阈值
        self.flip_horizontal = True  # 水平翻转开关（镜像模式）
        
        # 识别结果
        self.current_digit = None
        self.confidence = 0.0
        
        # 加载或训练模型
        self.load_or_train_model()
        
    def load_or_train_model(self):
        """加载已有模型或训练新模型"""
        if os.path.exists(self.model_path):
            print(f"加载已训练的模型: {self.model_path}")
            with open(self.model_path, 'rb') as f:
                self.recognizer = pickle.load(f)
            print("模型加载成功！")
        else:
            print("未找到已训练模型，开始训练新模型...")
            self.train_model()
            
    def train_model(self):
        """训练识别模型"""
        print("\n" + "=" * 60)
        print("训练数字识别模型")
        print("=" * 60)
        
        # 加载数据
        images, labels = load_mnist_data()
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.3, random_state=42
        )
        
        # 创建并训练识别器
        self.recognizer = DigitRecognizer(n_descriptors=30, classifier_type='svm')
        self.recognizer.train(X_train, y_train)
        
        # 评估性能
        accuracy, _ = self.recognizer.evaluate(X_test, y_test)
        print(f"\n模型训练完成！准确率: {accuracy:.4f}")
        
        # 保存模型
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.recognizer, f)
        print(f"模型已保存到: {self.model_path}")
        
    def preprocess_roi(self, roi):
        """
        预处理ROI区域
        
        Args:
            roi: 输入的ROI图像
            
        Returns:
            处理后的图像（黑底白字，和训练数据一致）
        """
        # 转灰度
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # 适度增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))  # 减小tile提高局部对比度
        enhanced = clahe.apply(gray)
        
        # 中等强度去噪（平衡噪声和细节）
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # 使用Otsu自动阈值 + 反转（更稳定）
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 去除小噪点
        kernel_denoise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_denoise, iterations=1)
        
        # 连接断开的笔画（闭运算）
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_connect, iterations=2)
        
        # 轻微膨胀加粗笔迹
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.dilate(binary, kernel_dilate, iterations=1)
        
        # 找到所有轮廓，保留所有有意义的部分
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # 创建空白画布
            mask = np.zeros_like(binary)
            
            # 计算所有轮廓的面积
            areas = [cv2.contourArea(c) for c in contours]
            max_area = max(areas) if areas else 0
            
            # 保留面积大于最大面积5%的所有轮廓（避免丢失笔画）
            for contour, area in zip(contours, areas):
                if area > max_area * 0.05:  # 保留较大的轮廓部分
                    cv2.drawContours(mask, [contour], -1, 255, -1)
            
            binary = mask
        
        # 调整大小到28x28
        resized = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_AREA)
        
        return resized
    
    def recognize_digit(self, roi):
        """
        识别ROI中的数字
        
        Args:
            roi: 输入的ROI图像
            
        Returns:
            识别的数字和置信度
        """
        # 预处理
        processed = self.preprocess_roi(roi)
        
        # 提取特征
        features = self.fd.extract_features(processed)
        
        # 预测
        if hasattr(self.recognizer.classifier, 'predict_proba'):
            # SVM等支持概率预测
            probabilities = self.recognizer.classifier.predict_proba([features])[0]
            digit = np.argmax(probabilities)
            confidence = probabilities[digit]
        else:
            # KNN等不支持概率，使用距离
            digit = self.recognizer.classifier.predict([features])[0]
            confidence = 0.5  # 默认置信度
        
        return digit, confidence, processed
    
    def draw_ui(self, frame, roi_rect, digit, confidence, processed_roi):
        """
        绘制用户界面
        
        Args:
            frame: 视频帧
            roi_rect: ROI矩形坐标
            digit: 识别的数字
            confidence: 置信度
            processed_roi: 处理后的ROI
        """
        h, w = frame.shape[:2]
        
        # 绘制ROI矩形框
        x, y, roi_w, roi_h = roi_rect
        cv2.rectangle(frame, (x, y), (x + roi_w, y + roi_h), (0, 255, 0), 2)
        
        # 绘制标题（英文，避免中文乱码）
        title = "Real-time Digit Recognition - Fourier Descriptor"
        cv2.putText(frame, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 绘制说明文字（英文）
        flip_status = "ON" if self.flip_horizontal else "OFF"
        instructions = [
            "Keys:",
            "  Q - Quit",
            "  S - Save result",
            "  C - Clear result",
            "  + - Increase threshold",
            "  - - Decrease threshold",
            f"  F - Flip mirror ({flip_status})",
            "",
            "Instructions:",
            "  1. Put digit in green box",
            "  2. Keep it clear",
            "  3. See result on right"
        ]
        
        y_offset = 60
        for line in instructions:
            cv2.putText(frame, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
        
        # 显示当前阈值和统计信息
        threshold_text = f"Threshold: {self.threshold_value}"
        cv2.putText(frame, threshold_text, (10, h - 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 显示处理后图像的统计信息（帮助调试）
        if processed_roi is not None and processed_roi.size > 0:
            white_pixels = np.sum(processed_roi > 128)
            total_pixels = processed_roi.size
            white_ratio = white_pixels / total_pixels * 100
            stat_text = f"White pixels: {white_ratio:.1f}%"
            cv2.putText(frame, stat_text, (10, h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 在右侧显示识别结果
        if digit is not None:
            # 绘制结果背景
            result_x = w - 250
            cv2.rectangle(frame, (result_x, 50), (w - 20, 350), (50, 50, 50), -1)
            cv2.rectangle(frame, (result_x, 50), (w - 20, 350), (255, 255, 255), 2)
            
            # 显示识别的数字（大字）
            digit_text = str(digit)
            text_size = cv2.getTextSize(digit_text, cv2.FONT_HERSHEY_SIMPLEX, 4, 8)[0]
            text_x = result_x + (230 - text_size[0]) // 2
            
            # 根据置信度设置颜色
            if confidence > 0.8:
                color = (0, 255, 0)  # 绿色 - 高置信度
            elif confidence > 0.5:
                color = (0, 255, 255)  # 黄色 - 中等置信度
            else:
                color = (0, 0, 255)  # 红色 - 低置信度
            
            cv2.putText(frame, digit_text, (text_x, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 4, color, 8)
            
            # 显示置信度
            conf_text = f"Confidence: {confidence*100:.1f}%"
            cv2.putText(frame, conf_text, (result_x + 20, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示处理后的图像
            if processed_roi is not None and processed_roi.size > 0:
                try:
                    # 放大到150x150，使用INTER_NEAREST保持清晰
                    display_roi = cv2.resize(processed_roi, (150, 150), 
                                            interpolation=cv2.INTER_NEAREST)
                    # 转为3通道
                    if len(display_roi.shape) == 2:
                        display_roi_color = cv2.cvtColor(display_roi, cv2.COLOR_GRAY2BGR)
                    else:
                        display_roi_color = display_roi
                    
                    # 放置到结果区域
                    frame[180:330, result_x+40:result_x+190] = display_roi_color
                    
                    # 绘制边框
                    cv2.rectangle(frame, (result_x+40, 180), (result_x+190, 330), 
                                (255, 255, 255), 2)
                    cv2.putText(frame, "Processed", (result_x + 60, 170),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                except Exception as e:
                    # 如果显示失败，显示错误信息
                    cv2.putText(frame, f"Error: {str(e)[:20]}", (result_x + 40, 250),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return frame
    
    def save_result(self, frame, digit, confidence):
        """保存识别结果"""
        timestamp = cv2.getTickCount()
        filename = f"recognized_digit_{digit}_conf{confidence*100:.0f}_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"✓ 结果已保存: {filename}")
    
    def run(self, camera_id=0):
        """
        运行实时识别系统
        
        Args:
            camera_id: 摄像头ID（默认0）
        """
        print("\n" + "=" * 60)
        print("启动实时数字识别系统")
        print("=" * 60)
        print("\n正在打开摄像头...")
        
        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("错误：无法打开摄像头！")
            print("请检查：")
            print("  1. 摄像头是否连接")
            print("  2. 摄像头是否被其他程序占用")
            print("  3. 尝试更换 camera_id (0, 1, 2...)")
            return
        
        # 设置摄像头分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✓ 摄像头已打开")
        print("\n按 Q 键退出\n")
        
        # ROI位置（居中）
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        roi_x = (frame_width - self.roi_size) // 2
        roi_y = (frame_height - self.roi_size) // 2
        
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("错误：无法读取摄像头画面")
                break
            
            # 镜像翻转（更符合直觉，按F键切换）
            if self.flip_horizontal:
                frame = cv2.flip(frame, 1)
            
            # 提取ROI
            roi = frame[roi_y:roi_y+self.roi_size, roi_x:roi_x+self.roi_size]
            
            # 识别数字
            digit, confidence, processed = self.recognize_digit(roi)
            
            # 绘制UI
            roi_rect = (roi_x, roi_y, self.roi_size, self.roi_size)
            frame = self.draw_ui(frame, roi_rect, digit, confidence, processed)
            
            # 显示画面
            cv2.imshow('Real-time Digit Recognition', frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                # 退出
                print("\n退出程序...")
                break
            elif key == ord('s') or key == ord('S'):
                # 保存结果
                if digit is not None:
                    self.save_result(frame, digit, confidence)
            elif key == ord('c') or key == ord('C'):
                # 清除结果
                digit = None
                confidence = 0.0
                print("已清除识别结果")
            elif key == ord('+') or key == ord('='):
                # 增加阈值
                self.threshold_value = min(255, self.threshold_value + 5)
                print(f"阈值: {self.threshold_value}")
            elif key == ord('-') or key == ord('_'):
                # 减少阈值
                self.threshold_value = max(0, self.threshold_value - 5)
                print(f"阈值: {self.threshold_value}")
            elif key == ord('f') or key == ord('F'):
                # 切换镜像翻转
                self.flip_horizontal = not self.flip_horizontal
                status = "开启" if self.flip_horizontal else "关闭"
                print(f"镜像翻转: {status}")
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("\n程序已退出")


def test_camera():
    """测试摄像头是否可用"""
    print("测试摄像头...")
    print("正在尝试打开摄像头 (ID=0)...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ 摄像头0打开失败")
        print("尝试摄像头1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("✗ 摄像头1打开失败")
            print("\n请检查：")
            print("  1. 摄像头是否正确连接")
            print("  2. 摄像头驱动是否安装")
            print("  3. 是否有其他程序占用摄像头")
            return False
        else:
            print("✓ 摄像头1可用")
            cap.release()
            return True
    else:
        print("✓ 摄像头0可用")
        # 测试读取一帧
        ret, frame = cap.read()
        if ret:
            print(f"✓ 摄像头分辨率: {frame.shape[1]}x{frame.shape[0]}")
        cap.release()
        return True


def main():
    """主函数"""
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + " " * 15 + "实时数字识别系统 - 基于傅里叶描述子" + " " * 15 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print("\n")
    
    # 测试摄像头
    if not test_camera():
        print("\n摄像头测试失败，程序退出")
        return
    
    print("\n" + "=" * 70)
    
    # 创建识别器
    recognizer = RealtimeDigitRecognizer()
    
    # 运行识别系统
    recognizer.run(camera_id=0)


if __name__ == "__main__":
    main()
