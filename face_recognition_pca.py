"""
PCA Face Recognition on ORL Dataset
完整的PCA人脸识别实验代码 - 修复版本
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. 数据加载
# ============================================================================
def load_orl_faces(data_path='orl_faces', test_size=0.2):
    """
    加载ORL人脸数据集 (s1-s40文件夹)
    
    Args:
        data_path: 数据集文件夹路径
        test_size: 测试集比例（20%）
    
    Returns:
        X_train, X_test, y_train, y_test: 训练和测试数据
        label_names: 标签名称 (s1-s40)
    """
    print("="*70)
    print("加载ORL人脸数据集...")
    print("="*70)
    
    images = []
    labels = []
    label_names = []
    
    # 获取所有人脸文件夹 (s1-s40)
    subject_dirs = sorted([d for d in os.listdir(data_path) 
                          if os.path.isdir(os.path.join(data_path, d))])
    
    label = 0
    for subject_dir in subject_dirs:
        subject_path = os.path.join(data_path, subject_dir)
        label_names.append(subject_dir)
        
        # 加载该人的所有人脸图像
        image_files = sorted([f for f in os.listdir(subject_path) 
                             if f.endswith(('.pgm', '.jpg', '.png'))])
        
        for image_file in image_files:
            image_path = os.path.join(subject_path, image_file)
            try:
                # 加载图像并转换为灰度
                img = Image.open(image_path).convert('L')
                img_array = np.array(img, dtype=np.float64).flatten()
                
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"加载错误 {image_path}: {e}")
        
        label += 1
    
    # 转换为numpy数组
    X = np.array(images, dtype=np.float64)
    y = np.array(labels, dtype=np.int32)
    
    print(f"✓ 加载了 {len(X)} 张图像，共 {len(subject_dirs)} 个人")
    print(f"✓ 图像大小（展平）: {X.shape}")
    
    # 按80-20比例分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # 确保数据类型正确
    X_train = np.ascontiguousarray(X_train, dtype=np.float64)
    X_test = np.ascontiguousarray(X_test, dtype=np.float64)
    y_train = np.ascontiguousarray(y_train, dtype=np.int32)
    y_test = np.ascontiguousarray(y_test, dtype=np.int32)
    
    print(f"✓ 训练集: {len(X_train)} 张图像 (80%)")
    print(f"✓ 测试集: {len(X_test)} 张图像 (20%)")
    print()
    
    return X_train, X_test, y_train, y_test, label_names


# ============================================================================
# 2. PCA降维
# ============================================================================
class PCADimensionalityReduction:
    """
    PCA (主成分分析) 用于降维
    
    原理：
    PCA通过寻找数据方差最大的正交方向（主成分），将高维数据投影到低维空间。
    保留最重要的特征信息，同时减少计算复杂度。
    """
    
    def __init__(self, n_components):
        """初始化PCA"""
        self.n_components = n_components
        self.pca = SklearnPCA(n_components=n_components, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X_train):
        """拟合PCA模型"""
        print(f"PCA拟合 (n_components={self.n_components})...")
        # 确保输入类型
        X_train = np.ascontiguousarray(X_train, dtype=np.float64)
        X_normalized = self.scaler.fit_transform(X_train)
        self.pca.fit(X_normalized)
        self.is_fitted = True
        
        # 打印解释的方差比
        total_variance = np.sum(self.pca.explained_variance_ratio_)
        print(f"  ✓ 解释方差比: {total_variance*100:.2f}%")
        return self
    
    def transform(self, X):
        """转换数据"""
        if not self.is_fitted:
            raise ValueError("PCA模型未拟合")
        X = np.ascontiguousarray(X, dtype=np.float64)
        X_normalized = self.scaler.transform(X)
        result = self.pca.transform(X_normalized)
        return np.ascontiguousarray(result, dtype=np.float64)
    
    def fit_transform(self, X):
        """拟合并转换"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """反向转换（重构）"""
        if not self.is_fitted:
            raise ValueError("PCA模型未拟合")
        X_transformed = np.ascontiguousarray(X_transformed, dtype=np.float64)
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        return self.scaler.inverse_transform(X_reconstructed)
    
    def get_explained_variance(self):
        """获取解释方差信息"""
        if not self.is_fitted:
            raise ValueError("PCA模型未拟合")
        variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratio)
        return variance_ratio, cumulative_variance
    
    def get_components(self):
        """获取主成分（特征脸）"""
        if not self.is_fitted:
            raise ValueError("PCA模型未拟合")
        return self.pca.components_


# ============================================================================
# 3. 分类器
# ============================================================================
class FaceClassifier:
    """人脸分类器"""
    
    def __init__(self, classifier_type='knn'):
        """
        初始化分类器
        
        Args:
            classifier_type: 'knn', 'svm', 或 'mlp'
        """
        self.classifier_type = classifier_type
        
        if classifier_type == 'knn':
            self.clf = KNeighborsClassifier(n_neighbors=5)
        elif classifier_type == 'svm':
            # 使用liblinear核函数，更快速稳定
            self.clf = SVC(kernel='rbf', C=1.0, gamma='auto')
        elif classifier_type == 'mlp':
            self.clf = MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                verbose=0,
                learning_rate='adaptive',
                learning_rate_init=0.001
            )
        else:
            raise ValueError(f"未知的分类器类型: {classifier_type}")
    
    def train(self, X_train, y_train):
        """训练分类器"""
        print(f"  训练{self.classifier_type.upper()}分类器...")
        # 确保数据类型正确
        X_train = np.ascontiguousarray(X_train, dtype=np.float64)
        y_train = np.ascontiguousarray(y_train, dtype=np.int32)
        
        start_time = time.time()
        self.clf.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"  ✓ 训练时间: {train_time:.4f}秒")
        return train_time
    
    def predict(self, X_test):
        """预测"""
        X_test = np.ascontiguousarray(X_test, dtype=np.float64)
        return self.clf.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """评估"""
        X_test = np.ascontiguousarray(X_test, dtype=np.float64)
        y_test = np.ascontiguousarray(y_test, dtype=np.int32)
        
        start_time = time.time()
        y_pred = self.predict(X_test)
        pred_time = time.time() - start_time
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy, y_pred, pred_time


# ============================================================================
# 4. 可视化
# ============================================================================
def plot_sample_faces(X_train, y_train, label_names, image_shape=(112, 92), n_per_class=4):
    """绘制样本人脸"""
    n_classes = min(8, len(np.unique(y_train)))
    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(12, n_classes * 1.5))
    
    for i in range(n_classes):
        indices = np.where(y_train == i)[0]
        for j in range(n_per_class):
            ax = axes[i, j]
            if j < len(indices):
                img = X_train[indices[j]].reshape(image_shape)
                ax.imshow(img, cmap='gray')
            ax.axis('off')
        axes[i, 0].set_ylabel(label_names[i], fontsize=9, fontweight='bold')
    
    plt.suptitle('ORL数据集样本人脸', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_misclassified_faces(X_test, y_test, y_pred, label_names, 
                            image_shape=(112, 92), n_faces=20):
    """绘制被错误分类的人脸"""
    # 找出错误分类的样本
    misclassified_idx = np.where(y_test != y_pred)[0]
    
    if len(misclassified_idx) == 0:
        print("    ✓ 没有错误分类的人脸！")
        return None
    
    n_show = min(n_faces, len(misclassified_idx))
    n_cols = 5
    n_rows = (n_show + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx < n_show:
            img_idx = misclassified_idx[idx]
            img = X_test[img_idx].reshape(image_shape)
            true_label = label_names[int(y_test[img_idx])]
            pred_label = label_names[int(y_pred[img_idx])]
            
            ax.imshow(img, cmap='gray')
            ax.set_title(f"正确: {true_label}\n预测: {pred_label}", 
                        color='red', fontsize=9, fontweight='bold')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle(f'被错误分类的人脸 (共{len(misclassified_idx)}张)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_accuracy_comparison(dimensions, accuracies_with_pca, 
                            accuracy_without_pca, classifier_names):
    """绘制精度对比"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for clf_name, color in zip(classifier_names, colors):
        accuracies = accuracies_with_pca[clf_name]
        ax.plot(dimensions, accuracies, marker='o', label=f'{clf_name} (with PCA)', 
               color=color, linewidth=2.5, markersize=8)
        
        # 添加不使用PCA的精度（作为水平线）
        baseline = accuracy_without_pca[clf_name]
        ax.axhline(y=baseline, color=color, linestyle='--', linewidth=2, alpha=0.7,
                  label=f'{clf_name} (no PCA)')
    
    ax.set_xlabel('PCA降维维度', fontsize=12, fontweight='bold')
    ax.set_ylabel('分类精度', fontsize=12, fontweight='bold')
    ax.set_title('PCA降维维度对人脸识别精度的影响', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xticks(dimensions)
    ax.set_ylim([0.6, 1.0])
    
    plt.tight_layout()
    return fig


def plot_eigenfaces(components, image_shape=(112, 92), n_faces=12):
    """绘制特征脸（主成分）"""
    n_show = min(n_faces, len(components))
    n_cols = 4
    n_rows = (n_show + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3))
    axes = axes.flatten()
    
    for idx in range(n_show):
        eigenface = components[idx].reshape(image_shape)
        # 归一化用于显示
        eigenface = (eigenface - eigenface.min()) / (eigenface.max() - eigenface.min() + 1e-10)
        
        axes[idx].imshow(eigenface, cmap='gray')
        axes[idx].set_title(f'特征脸 {idx+1}', fontsize=10)
        axes[idx].axis('off')
    
    for idx in range(n_show, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('PCA主成分（特征脸）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_variance_explained(pca_models, dimensions):
    """绘制解释的方差"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 单个成分的方差解释比
    for pca, dim in zip(pca_models, dimensions):
        variance_ratio, _ = pca.get_explained_variance()
        ax1.plot(range(1, min(51, len(variance_ratio) + 1)), variance_ratio[:50], 
                marker='o', label=f'{dim}D', linewidth=2)
    
    ax1.set_xlabel('主成分索引', fontsize=12, fontweight='bold')
    ax1.set_ylabel('解释方差比', fontsize=12, fontweight='bold')
    ax1.set_title('各主成分的方差解释比', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 累积解释方差
    for pca, dim in zip(pca_models, dimensions):
        _, cumulative_variance = pca.get_explained_variance()
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
                marker='o', label=f'{dim}D', linewidth=2)
    
    ax2.set_xlabel('主成分个数', fontsize=12, fontweight='bold')
    ax2.set_ylabel('累积解释方差比', fontsize=12, fontweight='bold')
    ax2.set_title('累积解释方差比', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig


# ============================================================================
# 5. 主实验流程
# ============================================================================
def main():
    """主函数"""
    
    # 创建结果文件夹
    os.makedirs('results', exist_ok=True)
    
    # ========== 加载数据 ==========
    X_train, X_test, y_train, y_test, label_names = load_orl_faces('orl_faces')
    
    # ========== 绘制样本 ==========
    print("\n绘制样本人脸...")
    fig_samples = plot_sample_faces(X_train, y_train, label_names)
    fig_samples.savefig('results/01_sample_faces.png', dpi=100, bbox_inches='tight')
    plt.close(fig_samples)
    print("✓ 已保存: results/01_sample_faces.png")
    
    # ========== 不使用PCA的基线 ==========
    print("\n" + "="*70)
    print("实验1: 不使用PCA的原始图像分类（基线）")
    print("="*70)
    
    classifier_names = ['KNN', 'SVM', 'MLP']
    accuracy_without_pca = {}
    
    for clf_name in classifier_names:
        print(f"\n使用{clf_name}分类器:")
        clf = FaceClassifier(clf_name.lower())
        train_time = clf.train(X_train, y_train)
        accuracy, y_pred, pred_time = clf.evaluate(X_test, y_test)
        accuracy_without_pca[clf_name] = accuracy
        print(f"  ✓ 分类精度: {accuracy*100:.2f}%")
        print(f"  ✓ 预测时间: {pred_time:.4f}秒")
    
    # ========== 使用PCA的分类 ==========
    print("\n" + "="*70)
    print("实验2: 使用不同维度PCA的分类")
    print("="*70)
    
    pca_dimensions = [30, 70, 100, 200]
    pca_models = []
    accuracies_with_pca = {clf_name: [] for clf_name in classifier_names}
    
    for dim in pca_dimensions:
        print(f"\n{'='*50}")
        print(f"PCA维度: {dim}")
        print(f"{'='*50}")
        
        # PCA拟合
        pca = PCADimensionalityReduction(dim)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        print(f"✓ PCA输出形状: {X_train_pca.shape}")
        pca_models.append(pca)
        
        # 分类
        for clf_name in classifier_names:
            print(f"\n  {clf_name}分类器:")
            clf = FaceClassifier(clf_name.lower())
            train_time = clf.train(X_train_pca, y_train)
            accuracy, y_pred, pred_time = clf.evaluate(X_test_pca, y_test)
            accuracies_with_pca[clf_name].append(accuracy)
            print(f"    ✓ 分类精度: {accuracy*100:.2f}%")
            print(f"    ✓ 预测时间: {pred_time:.4f}秒")
            
            # 保存第一次的错误分类图
            if dim == pca_dimensions[0] and clf_name == 'SVM':
                print(f"\n  绘制{dim}D PCA + {clf_name}的错误分类图...")
                fig_misc = plot_misclassified_faces(X_test_pca, y_test, y_pred, label_names)
                if fig_misc:
                    fig_misc.savefig(f'results/02_misclassified_PCA{dim}_{clf_name}.png', 
                                    dpi=100, bbox_inches='tight')
                    plt.close(fig_misc)
                    print(f"  ✓ 已保存: results/02_misclassified_PCA{dim}_{clf_name}.png")
    
    # ========== 绘制特征脸 ==========
    print("\n\n绘制PCA特征脸...")
    fig_eigen = plot_eigenfaces(pca_models[0].get_components(), n_faces=12)
    fig_eigen.savefig('results/03_eigenfaces_PCA30.png', dpi=100, bbox_inches='tight')
    plt.close(fig_eigen)
    print("✓ 已保存: results/03_eigenfaces_PCA30.png")
    
    # ========== 绘制方差解释 ==========
    print("\n绘制方差解释图...")
    fig_var = plot_variance_explained(pca_models, pca_dimensions)
    fig_var.savefig('results/04_variance_explained.png', dpi=100, bbox_inches='tight')
    plt.close(fig_var)
    print("✓ 已保存: results/04_variance_explained.png")
    
    # ========== 绘制精度对比 ==========
    print("\n绘制精度对比图...")
    fig_acc = plot_accuracy_comparison(pca_dimensions, accuracies_with_pca, 
                                       accuracy_without_pca, classifier_names)
    fig_acc.savefig('results/05_accuracy_comparison.png', dpi=100, bbox_inches='tight')
    plt.close(fig_acc)
    print("✓ 已保存: results/05_accuracy_comparison.png")
    
    # ========== 生成结果报告 ==========
    print("\n" + "="*70)
    print("生成结果报告...")
    print("="*70)
    
    report_path = 'results/REPORT.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("PCA人脸识别实验报告\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. 实验概述\n")
        f.write("-" * 70 + "\n")
        f.write("数据集: ORL Face Dataset\n")
        f.write(f"总图像数: {len(X_train) + len(X_test)}\n")
        f.write(f"训练集: {len(X_train)}张 (80%)\n")
        f.write(f"测试集: {len(X_test)}张 (20%)\n")
        f.write(f"原始特征维度: {X_train.shape[1]}\n")
        f.write(f"PCA降维维度: {pca_dimensions}\n")
        f.write(f"分类器: {classifier_names}\n\n")
        
        f.write("2. 不使用PCA的基线结果\n")
        f.write("-" * 70 + "\n")
        for clf_name in classifier_names:
            f.write(f"{clf_name:10s}: {accuracy_without_pca[clf_name]*100:6.2f}%\n")
        f.write("\n")
        
        f.write("3. 使用PCA后的分类精度对比\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'维度':<10}")
        for clf_name in classifier_names:
            f.write(f"{clf_name:<15}")
        f.write("\n")
        f.write("-" * 70 + "\n")
        
        for i, dim in enumerate(pca_dimensions):
            f.write(f"{dim:<10}")
            for clf_name in classifier_names:
                acc = accuracies_with_pca[clf_name][i]
                f.write(f"{acc*100:6.2f}%{'':<8}")
            f.write("\n")
        f.write("\n")
        
        # 计算精度改进
        f.write("4. PCA与无PCA的精度差异 (+ 表示改进)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'维度':<10}")
        for clf_name in classifier_names:
            f.write(f"{clf_name:<15}")
        f.write("\n")
        f.write("-" * 70 + "\n")
        
        for i, dim in enumerate(pca_dimensions):
            f.write(f"{dim:<10}")
            for clf_name in classifier_names:
                diff = accuracies_with_pca[clf_name][i] - accuracy_without_pca[clf_name]
                sign = "+" if diff >= 0 else ""
                f.write(f"{sign}{diff*100:5.2f}%{'':<8}")
            f.write("\n")
        f.write("\n")
        
        # 最佳结果
        f.write("5. 分析与结论\n")
        f.write("-" * 70 + "\n")
        
        # 找到最佳精度
        best_accuracy = 0
        best_clf = ""
        best_dim = 0
        
        for clf_name in classifier_names:
            for i, dim in enumerate(pca_dimensions):
                if accuracies_with_pca[clf_name][i] > best_accuracy:
                    best_accuracy = accuracies_with_pca[clf_name][i]
                    best_clf = clf_name
                    best_dim = dim
        
        f.write(f"\n✓ 最佳结果: {best_clf}分类器 + {best_dim}D PCA\n")
        f.write(f"  精度: {best_accuracy*100:.2f}%\n")
        f.write(f"  降维比例: {best_dim}/{X_train.shape[1]} = {best_dim/X_train.shape[1]*100:.2f}%\n\n")
        
        f.write("✓ PCA降维原理:\n")
        f.write("  1) 计算数据的协方差矩阵\n")
        f.write("  2) 提取特征值和特征向量\n")
        f.write("  3) 选择最大的n个特征值对应的特征向量\n")
        f.write("  4) 将数据投影到这些特征向量构成的子空间\n\n")
        
        f.write("✓ PCA降维优势:\n")
        f.write(f"  1) 显著降低计算复杂度 (从{X_train.shape[1]}维到{best_dim}维)\n")
        f.write(f"  2) 减少存储空间需求 (降低到 {best_dim/X_train.shape[1]*100:.1f}%)\n")
        f.write(f"  3) 加快分类速度\n")
        f.write(f"  4) 去除噪声和冗余特征\n")
        f.write(f"  5) 保留数据最重要的信息\n\n")
        
        f.write("✓ 维度选择分析:\n")
        f.write(f"  30D:  计算最快，精度: {accuracies_with_pca[best_clf][0]*100:.2f}%\n")
        f.write(f"  70D:  平衡方案，精度: {accuracies_with_pca[best_clf][1]*100:.2f}%\n")
        f.write(f"  100D: 推荐方案，精度: {accuracies_with_pca[best_clf][2]*100:.2f}%\n")
        f.write(f"  200D: 最高精度，精度: {accuracies_with_pca[best_clf][3]*100:.2f}%\n\n")
        
        f.write("✓ 最优方案:\n")
        f.write(f"  综合考虑精度和效率，推荐使用:\n")
        f.write(f"  - 降维维度: 100D (可兼顾精度和复杂度)\n")
        f.write(f"  - 分类器: {best_clf}\n")
        f.write(f"  - 降维比例: 100D/{X_train.shape[1]} = {100/X_train.shape[1]*100:.1f}%\n")
        f.write(f"  - 期望精度: ~{accuracies_with_pca[best_clf][2]*100:.2f}%\n")
    
    print(f"✓ 已保存: {report_path}")
    
    # 打印报告
    print("\n" + "="*70)
    with open(report_path, 'r', encoding='utf-8') as f:
        print(f.read())
    
    print("\n" + "="*70)
    print("所有结果已保存到 results/ 文件夹")
    print("="*70)
    print("包括:")
    print("  ✓ 01_sample_faces.png - ORL数据集样本人脸")
    print("  ✓ 02_misclassified_*.png - 被错误分类的人脸")
    print("  ✓ 03_eigenfaces_PCA30.png - PCA特征脸（主成分）")
    print("  ✓ 04_variance_explained.png - 方差解释分析图")
    print("  ✓ 05_accuracy_comparison.png - 分类精度对比")
    print("  ✓ REPORT.txt - 详细实验报告")
    print("="*70)


if __name__ == '__main__':
    main()
