import numpy as np
from scipy.spatial import KDTree
class ICP:
    def __init__(self, max_iterations=100, tolerance=1e-6, 
                 sampling_rate=1.0, use_scale=False):
        """
        ICP算法的完整实现类
        
        参数:
        max_iterations: 最大迭代次数
        tolerance: 收敛阈值
        sampling_rate: 采样率 (0.0-1.0)
        use_scale: 是否估计尺度
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.sampling_rate = sampling_rate
        self.use_scale = use_scale
        
        self.R = None
        self.t = None
        self.s = 1.0  # 尺度因子
        self.errors = []
        
    def subsample_points(self, points):
        """随机子采样点云"""
        n_points = len(points)
        n_samples = int(n_points * self.sampling_rate)
        indices = np.random.choice(n_points, n_samples, replace=False)
        return points[indices]
    
    def compute_transform(self, source, target):
        """使用SVD计算最优变换"""
        # 计算重心
        source_centroid = np.mean(source, axis=0)
        target_centroid = np.mean(target, axis=0)
        
        # 中心化
        source_centered = source - source_centroid
        target_centered = target - target_centroid
        
        # 计算协方差矩阵
        H = source_centered.T @ target_centered
        
        # SVD分解
        U, S, Vt = np.linalg.svd(H)
        
        # 计算旋转矩阵
        R = Vt.T @ U.T
        
        # 确保右手坐标系
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # 计算平移
        if self.use_scale:
            # 估计尺度
            self.s = np.trace(Vt.T @ np.diag(S) @ U.T) / np.sum(np.linalg.norm(source_centered, axis=1) ** 2)
            t = target_centroid - self.s * R @ source_centroid
        else:
            t = target_centroid - R @ source_centroid
            
        return R, t
    
    def register(self, source, target, initial_transform=None):
        """
        执行ICP配准
        
        参数:
        source: 源点云 (N x 3)
        target: 目标点云 (M x 3)
        initial_transform: 初始变换 (4x4矩阵)
        
        返回:
        transformed_source: 变换后的源点云
        transformation: 4x4变换矩阵
        """
        # 初始化
        source = source.copy().astype(np.float64)
        target = target.copy().astype(np.float64)
        
        if initial_transform is not None:
            # 应用初始变换
            source_hom = np.hstack([source, np.ones((len(source), 1))])
            source = (initial_transform @ source_hom.T).T[:, :3]
        
        # 创建KDTree用于最近邻搜索
        target_tree = KDTree(target)
        
        # 初始化变换
        self.R = np.eye(2)
        self.t = np.zeros(2)
        self.s = 1.0
        self.errors = []
        
        for iteration in range(self.max_iterations):
            # 子采样
            if self.sampling_rate < 1.0:
                source_sampled = self.subsample_points(source)
            else:
                source_sampled = source
            
            # 找到最近点
            distances, indices = target_tree.query(source_sampled)
            
            # 计算误差
            mean_error = np.mean(distances)
            self.errors.append(mean_error)
            
            # 检查收敛
            if iteration > 0 and abs(self.errors[-2] - mean_error) < self.tolerance:
                print(f"在第 {iteration+1} 次迭代后收敛")
                break
            
            # 获取对应点
            target_correspondences = target[indices]
            
            # 计算最优变换
            R_current, t_current = self.compute_transform(source_sampled, target_correspondences)
            
            # 更新累积变换
            if self.use_scale:
                self.R = R_current @ self.R
                self.t = R_current @ self.t + t_current
                source = (self.s * R_current @ source.T + t_current.reshape(-1, 1)).T
            else:
                self.R = R_current @ self.R
                self.t = R_current @ self.t + t_current
                source = (R_current @ source.T + t_current.reshape(-1, 1)).T
            
            print(f"迭代 {iteration+1}, 误差: {mean_error:.6f}")
        
        # 计算最终变换矩阵
        if self.use_scale:
            transformation = np.eye(3)
            transformation[:2, :2] = self.s * self.R
            transformation[:2, 2] = self.t
        else:
            transformation = np.eye(3)
            transformation[:2, :2] = self.R
            transformation[:2, 2] = self.t
        
        return source, transformation
    
    def plot_convergence(self):
        """绘制收敛曲线"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.errors, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('迭代次数')
        plt.ylabel('平均误差')
        plt.title('ICP收敛曲线')
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def project_points(origin_points, transformation):
        Rot = transformation[:2,:2]
        Trans = transformation[:2,2]
        return (Rot @ origin_points.T + Trans.reshape(-1, 1)).T
