# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1. # ndim 维度 4; dt(delta time) 时间步长 1  

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim) # 系统矩阵(状态转移矩阵)A [] 2ndim x 2ndim 单位矩阵
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt # 系统矩阵(状态转移矩阵)A 前ndim行的第ndim+i列元素设为dt -> 这意味着这些状态变量在每个时间步长中都会根据时间步长dt进行更新
        self._update_mat = np.eye(ndim, 2 * ndim) # 状态更新矩阵 -> 描述了如何从测量中提取与状态变量相关的信息

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20 # 位置权重系数, 用于控制模型中位置估计的不确定性
        self._std_weight_velocity = 1. / 160 # 速度权重系数, 用于控制模型中速度估计的不确定性

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement # 将测量值作为位置的均值（mean_pos）
        mean_vel = np.zeros_like(mean_pos) # 创建一个与均值位置（mean_pos）具有相同形状的全零数组，作为速度的均值（mean_vel）
        mean = np.r_[mean_pos, mean_vel] # 将位置均值和速度均值按行连接起来，形成一个包含位置和速度的均值向量

        std = [ # 根据测量值的高度（measurement[3]）和预先定义的位置和速度权重系数，计算了一个包含位置和速度的标准差列表
            2 * self._std_weight_position * measurement[3], # 2倍的位置权重系数乘以测量高度
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]] # 10倍的速度权重系数乘以测量高度
        covariance = np.diag(np.square(std)) # 协方差 根据标准差列表，创建一个对角矩阵，每个对角元素为对应标准差的平方.这个对角矩阵表示位置和速度之间的协方差。
        return mean, covariance # 返回均值向量（mean）和协方差矩阵（covariance）作为新跟踪目标的初始估计

    def predict(self, mean, covariance): # 预测 -> 时间更新
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [ # 根据预先定义的位置权重系数和上一时刻状态的高度（mean[3]），计算了一个包含位置维度标准差的列表
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [ # 根据预先定义的速度权重系数和上一时刻状态的高度（mean[3]），计算了一个包含速度维度标准差的列表
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])) # 过程噪声w的协方差Q 根据位置和速度维度的标准差.创建一个对角矩阵, 每个对角元素为对应标准差的平方.

        #mean = np.dot(self._motion_mat, mean)
        mean = np.dot(mean, self._motion_mat.T) # xA.T 将上一时刻状态的均值向量与系统动态矩阵的转置相乘，得到状态预测均值向量
        covariance = np.linalg.multi_dot(( # 先验误差协方差
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov # 先验误差协方差 APA.T + Q

        return mean, covariance # 返回状态的预测均值向量（mean）和协方差矩阵（covariance）

    def project(self, mean, covariance): # 
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [ # 标准差列表
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std)) # 根据测量空间维度的标准差，创建一个对角矩阵，每个对角元素为对应标准差的平方。这个对角矩阵表示状态在测量空间的噪声协方差

        mean = np.dot(self._update_mat, mean) # 通过矩阵乘法，将状态的均值向量与状态更新矩阵相乘，得到状态在测量空间的投影均值向量
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T)) # 计算状态在测量空间的投影协方差矩阵 
        return mean, covariance + innovation_cov # 返回状态在测量空间的投影均值向量（mean）和协方差矩阵（covariance）加上测量空间噪声协方差（innovation_cov）

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T # 根据位置和速度维度的标准差，创建一个二维数组，每一行对应一个对象的状态预测阶段的运动噪声协方差

        motion_cov = [] # 运动协方差
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i])) # 将运动协方差构造为对角函数并存储到列表中
        motion_cov = np.asarray(motion_cov) # 转化为numpy数组

        mean = np.dot(mean, self._motion_mat.T) # 将所有对象的状态均值矩阵与系统动态矩阵的转置相乘，得到状态的预测均值矩阵
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2)) # AP 通过矩阵乘法，将系统动态矩阵与状态协方差矩阵相乘，并使用 transpose 函数来调整数组的维度顺序
        covariance = np.dot(left, self._motion_mat.T) + motion_cov # 计算状态的预测协方差矩阵 APA.T + Q

        return mean, covariance # 回状态的预测均值矩阵（mean）和协方差矩阵（covariance）

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance) # 获取预测后的均值和协方差

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False) # 使用 Cholesky 分解对投影后的协方差进行分解，得到 Cholesky 因子（chol_factor）和下三角矩阵标志（lower）
        kalman_gain = scipy.linalg.cho_solve( # 计算修正矩阵 -> 卡尔曼增益
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T # 使用 Cholesky 分解的结果和预测的协方差矩阵，通过矩阵乘法计算卡尔曼增益
        innovation = measurement - projected_mean # 计算测量值与投影后的均值之间的差

        new_mean = mean + np.dot(innovation, kalman_gain.T) # 更新观测值
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T)) # 更新误差协方差
        return new_mean, new_covariance # 返回更新后的状态均值矩阵和协方差矩阵

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance) # 调用 project 方法对状态分布的均值和协方差进行投影，得到投影后的均值（mean）和协方差（covariance）
        if only_position: # 如果 only_position 参数为 True，则仅计算与边界框中心位置相关的距离
            mean, covariance = mean[:2], covariance[:2, :2] # 在这种情况下，将状态分布的均值和协方差限制在前两个维度(x 和 y)
            measurements = measurements[:, :2] # 同时将测量矩阵限制在前两列（x 和 y）

        d = measurements - mean # 计算测量值与状态分布均值之间的差异
        if metric == 'gaussian': # 如果距离度量标准为 'gaussian'
            return np.sum(d * d, axis=1) # 则计算欧氏距离的平方，即每个测量与状态分布之间的差异平方和
        elif metric == 'maha': # 如果距离度量标准为 'maha', 则计算马氏距离的平方
            cholesky_factor = np.linalg.cholesky(covariance) # 使用 Cholesky 分解对投影后的协方差进行分解得到 Cholesky 因子
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True) # 通过解三角线性方程组计算变换后的差异向量
            squared_maha = np.sum(z * z, axis=0) # 计算变换后的差异向量的平方和作为马氏距离的平方
            return squared_maha
        else:
            raise ValueError('invalid distance metric') # 如果距离度量标准不是 'gaussian' 或 'maha'，则抛出 ValueError 异常