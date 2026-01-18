[TOC]

# Week 30: 机器学习补遗：时序信号处理与数学特征工程

## 摘要

本周重点了解了几种基于信号处理和统计学的时序预处理方法，包括离散小波变换、卡尔曼滤波和分数阶差分，均拥有完备的数学理论支撑。通过数学推导，理解这些方法如何在保留信号有效信息的同时，去除噪声并实现平稳性。

## Abstract

This week's focus has been on several time-series preprocessing methods grounded in signal processing and statistics, including discrete wavelet transform, Kalman filtering, and fractional differencing. Each method is underpinned by a robust mathematical foundation. Through mathematical derivation, we have gained insight into how these techniques preserve the signal's effective information while removing noise and achieving stationarity.

## 1. 离散小波变换 (DWT)

### 1.1 理论背景

传统的傅里叶变换 (Fourier Transform) 将信号分解为正弦波的叠加：
$$
F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} dt
$$
傅里叶变换在频域上是局部的，但在时域上是全局的。这意味着它无法告诉我们“某个频率在什么时候发生”，因此不适合处理非平稳（频率随时间变化）的时序数据。

小波变换 （Wavelet Transform）引入了时频局部化的概念。它通过缩放（Scale, $a$）和平移（Translation, $\tau$）一个母小波函数 $\psi(t)$ 来分解信号。

对于离散序列，通过使用 Mallat 算法（多分辨率分析, MRA），信号被分解为：
1.  近似系数 （Approximation, $cA$）：捕捉信号的低频、宏观趋势。
2.  细节系数 （Detail, $cD$）：捕捉信号的高频、噪声或突变。

数学上，这是通过一组正交滤波器组实现的：
$$
\begin{aligned}
y_{low}[n] &= \sum_{k=-\infty}^{\infty} x[k] h[2n-k]\\
y_{high}[n] &= \sum_{k=-\infty}^{\infty} x[k] g[2n-k]
\end{aligned}
$$
其中 $h[n]$ 是低通滤波器，$g[n]$ 是高通滤波器，两者满足正交性关系。

### 1.2 阈值收缩

DWT 的核心优势在于去噪，基于假设：真实信号主要集中在大幅值的系数中，而噪声分布在所有系数中且幅值较小。

我们采用软阈值函数（Soft Thresholding）对高频细节系数 $cD$ 进行处理：
$$
\eta_\lambda(x) = \text{sign}(x) \cdot \max(|x| - \lambda, 0)
$$
其中 $\lambda$ 是阈值（通常由 Donoho-Johnstone 准则 $\lambda = \sigma \sqrt{2 \ln N}$ 确定），这在数学上等价于在一个 Besov 空间中进行平滑，具有极强的可解释性。

### 1.3 代码实现

```python
import numpy as np
import pywt
import matplotlib.pyplot as plt

def wavelet_denoising(data, wavelet='db4', level=1):
    """
    使用小波变换进行去噪
    Args:
        data: 输入时间序列
        wavelet: 母小波名称 (如 Daubechies 4)
        level: 分解层数
    """
    # 1. 分解 (Decomposition)
    # coeff[0] 是近似系数 cA，coeff[1:] 是各层的细节系数 cD
    coeff = pywt.wavedec(data, wavelet, mode="per", level=level)
    
    # 2. 计算阈值 (Universal Threshold)
    # 使用最底层细节系数的中位数估计噪声标准差 sigma
    sigma = np.median(np.abs(coeff[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    
    # 3. 阈值处理 (Thresholding)
    # 仅对细节系数 (High Frequency) 进行收缩，保留近似系数 (Trend)
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:])
    
    # 4. 重构 (Reconstruction)
    return pywt.waverec(coeff, wavelet, mode='per')

# 示例调用
# clean_signal = wavelet_denoising(noisy_signal, level=2)
```

## 2. 卡尔曼滤波 (Kalman Filter)

### 2.1 最优递归估计

卡尔曼滤波是一种针对线性动态系统的递归滤波器，它能够在存在测量噪声的情况下，对系统的内部状态进行最小均方误差 （Minimum Mean Square Error, MMSE）估计。

假设系统由以下方程描述：
1.  状态方程（State Transition）：
    $$ x_k = F x_{k-1} + B u_k + w_k $$
    其中 $x_k$ 是真实状态（如真实价格趋势），$w_k \sim \mathcal{N}(0, Q)$ 是过程噪声。
2.  观测方程（Measurement）：
    $$ z_k = H x_k + v_k $$
    其中 $z_k$ 是观测值（如市场报价，含噪声），$v_k \sim \mathcal{N}(0, R)$ 是测量噪声。

### 2.2 核心机制

卡尔曼滤波的精髓在于“卡尔曼增益 ($K$)”，它动态权衡“模型预测”与“传感器观测”的可信度。

1. 预测阶段（Predict）：基于上一时刻状态预测当前时刻。
    $$
    \begin{aligned}
    \hat{x}_{k}^- &= F \hat{x}_{k-1} \\
    P_{k}^- &= F P_{k-1} F^T + Q \quad \text{(协方差传播)}
    \end{aligned}
    $$

2. 更新阶段（Update）：利用观测值修正预测。
    $$
    \begin{aligned}
    K_k &= P_k^- H^T (H P_k^- H^T + R)^{-1} \quad \text{(卡尔曼增益)} \\
    \hat{x}_k &= \hat{x}_k^- + K_k (z_k - H \hat{x}_k^-) \quad \text{(状态修正)} \\
    P_k &= (I - K_k H) P_k^- \quad \text{(协方差收敛)}
    \end{aligned}
    $$

数学直觉：
*   当测量噪声 $R$ 很大时，增益 $K$ 变小，滤波器更相信模型预测 $x_k^-$（平滑效果强）。
*   当过程噪声 $Q$ 很大时，增益 $K$ 变大，滤波器更相信观测值 $z_k$（响应速度快）。

### 2.3 代码实现

```python
class SimpleKalmanFilter:
    def __init__(self, initial_value, process_noise=1e-5, measure_noise=1e-1):
        """
        一维卡尔曼滤波器，用于平滑时序数据
        假设 F=1, H=1 (随机游走模型)
        """
        self.x = initial_value  # 状态估计
        self.P = 1.0            # 估计协方差 (不确定性)
        self.Q = process_noise  # 过程噪声协方差
        self.R = measure_noise  # 测量噪声协方差
        
    def update(self, measurement):
        # 1. 预测 (Predict)
        # x_pred = x_prev (随机游走假设)
        # P_pred = P_prev + Q
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # 2. 更新 (Update)
        # 计算卡尔曼增益 K
        K = P_pred / (P_pred + self.R)
        
        # 更新状态估计
        self.x = x_pred + K * (measurement - x_pred)
        
        # 更新协方差
        self.P = (1 - K) * P_pred
        
        return self.x

# 应用示例
# kf = SimpleKalmanFilter(prices[0])
# smoothed = [kf.update(z) for z in prices]
```

## 3. 平稳性与记忆性的平衡：分数阶差分

### 3.1 理论背景

在处理非平稳时序数据时，我们通常使用一阶差分（Differencing, $d=1$）使其平稳：
$$
\Delta x_t = x_t - x_{t-1}
$$
这虽然去除了趋势，实现了平稳性，但也完全抹除了长期的记忆信息。这对于需要捕捉长期依赖的 Transformer 或 LSTM 模型是毁灭性的。

分数阶差分 (Fractional Differentiation) 允许差分阶数 $d$ 为小数（例如 $0 < d < 1$），试图在“平稳性”和“记忆性”之间找到最佳平衡点。

### 3.2 数学推导

根据滞后算子 (Lag Operator, $L$) 的定义，其中 $L x_t = x_{t-1}$。
差分运算可以表示为 $(1-L)^d x_t$。
对 $(1-L)^d$ 进行广义二项式展开（Maclaurin级数）：
$$
(1-L)^d = \sum_{k=0}^{\infty} \binom{d}{k} (-1)^k L^k = \sum_{k=0}^{\infty} \omega_k L^k
$$

权重 $\omega_k$ 的递归计算公式为：
$$
\omega_0 = 1, \quad \omega_k = - \omega_{k-1} \frac{d - k + 1}{k}
$$

当 $d$ 为整数时，权重迅速归零（有限脉冲响应）。当 $d$ 为分数时，权重 $\omega_k$ 缓慢衰减。这意味着当前的值 $x_t$ 是过去无限长历史数据的加权和，从而保留了历史记忆。

### 3.3 代码实现

```python
def get_weights_frac_diff(d, size):
    """
    计算分数阶差分的权重系数
    Args:
        d: 差分阶数 (如 0.4)
        size: 截断窗口大小
    """
    w = [1.]
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1) / k
        w.append(w_k)
    return np.array(w[::-1]) # 翻转以匹配卷积方向

def fractional_diff(series, d=0.4, threshold=1e-5):
    """
    对时间序列应用分数阶差分
    """
    # 动态确定窗口大小，直到权重衰减到忽略不计
    weights = get_weights_frac_diff(d, len(series))
    
    # 使用卷积计算差分结果 (加权求和)
    # Valid模式会损失前几个数据点
    res = np.convolve(series, weights, mode='valid')
    return res

# 效果：res 保留了更多的 trend 信息，同时方差比原始数据更稳定
```

## 总结

本周我们从纯模型的构建暂时抽离，通过严谨的数学推导重新审视了数据预处理环节。小波变换利用正交基函数的多分辨率特性，在不损失时域信息的前提下分离了信号与噪声，优于传统的傅里叶变换。而卡尔曼滤波建立在状态空间与贝叶斯估计之上，为含噪数据提供了一种最优的平滑方案，特别适合流式数据处理。这些预处理方法不仅提高了数据的信噪比，更重要的是，它们为后续的模型提供了分布更规范、特征更显著的输入，从而加速收敛并提升泛化能力。
