[TOC]

# Week 27: 机器学习补遗：XGBoost

## 摘要

本周继续回归经典机器学习领域，对XGBoost 进行了学习。本周重点推导了 XGBoost 基于二阶泰勒展开的目标函数，并结合时序预测场景进行学习。

## Abstract

This week we continued our return to the classic field of machine learning, focusing on XGBoost. The primary emphasis was on deriving XGBoost's objective function based on a second-order Taylor expansion, alongside its application within a time series forecasting scenario.

## 1. 数学原理：二阶泰勒展开与目标函数

### 1.1 概要

XGBoost (Extreme Gradient Boosting) 的核心优势在于其对目标函数的二阶近似和显式的正则化项。与传统 GBDT 只利用一阶导数不同，XGBoost 利用了泰勒展开保留了二阶导数信息，使得目标函数的下降更加精准。

设 $y_i$ 为真实值，$\hat{y}_i^{(t)}$ 为第 $t$ 轮的预测值。目标函数定义为：
$$
\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)
$$
其中 $f_t(x_i)$ 是第 $t$ 棵树的预测结果，$\Omega(f_t)$ 是正则化项。

对损失函数 $l$ 进行二阶泰勒展开：
$$
\mathcal{L}^{(t)} \approx \sum_{i=1}^n [l(y_i, \hat{y}^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)] + \Omega(f_t)
$$
其中 $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ 为一阶梯度，$h_i = \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ 为二阶梯度。由于 $l(y_i, \hat{y}^{(t-1)})$ 是常数，优化目标简化为最小化：
$$
\tilde{\mathcal{L}}^{(t)} = \sum_{i=1}^n [g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)] + \Omega(f_t)
$$

### 1.2 代码实现

```python
import numpy as np
import xgboost as xgb
from typing import Tuple

def custom_logloss_objective(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    """
    手动实现基于LogLoss的一阶(grad)和二阶(hess)梯度计算
    Preds: 模型的原始输出 (Logits)，未经过Sigmoid
    """
    labels = dtrain.get_label()
    
    # Sigmoid变换: 1 / (1 + exp(-x))
    preds_prob = 1.0 / (1.0 + np.exp(-preds))
    
    # 一阶梯度 grad = p - y
    grad = preds_prob - labels
    
    # 二阶梯度 hess = p * (1 - p)
    hess = preds_prob * (1.0 - preds_prob)
    
    return grad, hess

# 使用示例
# model = xgb.train(params, dtrain, obj=custom_logloss_objective)
```

### 1.3 效果分析

通过引入二阶导数 $h_i$，XGBoost 相当于在优化过程中利用了牛顿法（Newton's Method）的思想，比单纯的一阶梯度下降收敛速度更快。同时，自定义目标函数赋予了模型极强的灵活性，能够适应不对称损失等复杂场景。

## 2. 结构打分与正则化

### 2.1 概要

为了防止过拟合，XGBoost 将树的复杂度显式加入目标函数：
$$
\Omega(f) = \gamma T + \frac{1}{2}\lambda ||w||^2
$$
其中 $T$ 是叶子节点数量，$w$ 是叶子节点的权重向量。

将所有样本按照叶子节点归组，令 $I_j = \{i|q(x_i)=j\}$ 为属于第 $j$ 个叶子节点的样本集合。目标函数可重写为关于叶子权重 $w_j$ 的一元二次方程。求解后，可得第 $t$ 棵树的最优结构分数（Structure Score）：

$$
Obj^* = -\frac{1}{2} \sum_{j=1}^T \frac{(\sum_{i\in I_j} g_i)^2}{\sum_{i\in I_j} h_i + \lambda} + \gamma T
$$
这个分数类似于决策树中的 Gini 系数或信息增益，用于评价树结构的优劣。分裂收益（Gain）计算公式为：
$$
Gain = \frac{1}{2} \left[ \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right] - \gamma
$$

### 2.2 代码实现

```python
class XGBNodeSplitter:
    def __init__(self, reg_lambda=1.0, gamma=0.0):
        self.reg_lambda = reg_lambda
        self.gamma = gamma

    def calculate_score(self, G, H):
        """
        计算单个叶子节点的分数
        G: 该节点所有样本的一阶梯度之和
        H: 该节点所有样本的二阶梯度之和
        """
        return -0.5 * (G**2) / (H + self.reg_lambda)

    def calculate_split_gain(self, G_left, H_left, G_right, H_right):
        """
        计算分裂增益 Gain
        """
        # 分裂前的梯度和
        G_total = G_left + G_right
        H_total = H_left + H_right
        
        # 分裂前的分数 (Root)
        score_before = self.calculate_score(G_total, H_total) # 注意：公式中常省略前面的负号用于求最大值，这里保持一致
        # 但Gain公式通常是：Split后分数和 - Split前分数 - 代价
        
        term_L = (G_left**2) / (H_left + self.reg_lambda)
        term_R = (G_right**2) / (H_right + self.reg_lambda)
        term_Total = (G_total**2) / (H_total + self.reg_lambda)
        
        gain = 0.5 * (term_L + term_R - term_Total) - self.gamma
        return gain

# 模拟数据
splitter = XGBNodeSplitter(reg_lambda=1.0, gamma=0.1)
gain = splitter.calculate_split_gain(G_left=10, H_left=5, G_right=20, H_right=8)
print(f"Split Gain: {gain:.4f}")
```

### 2.3 关键特性分析

公式中的 $\lambda$ 充当了平滑项，当某个叶子节点样本极少（$H$ 很小）时，$\lambda$ 避免了权重 $w$ 过大，起到了抑制过拟合的作用。而 $\gamma$ 则是分裂的“最低门槛”，只有当分裂带来的增益大于 $\gamma$ 时，节点才会分裂，这相当于自动进行了预剪枝（Pre-pruning）。

## 3. 时序预测中的应用

### 3.1 概要

在时序预测中，XGBoost 不能像 RNN/Transformer 那样直接处理序列依赖，需要将时序问题转化为监督学习问题（Supervised Learning）。核心在于**特征工程**，即将时间步 $t$ 的预测依赖转化为 $t-1, t-2...$ 的特征输入。

### 3.2 代码实现

```python
import pandas as pd

def create_time_series_features(df: pd.DataFrame, target_col: str, lags: list, window: int):
    """
    构造时序特征：Lag特征 和 滚动统计特征
    """
    df_feat = df.copy()
    
    # 1. 构造滞后特征 (Lag Features)
    # 捕捉原本的序列依赖：相当于 AR 模型
    for lag in lags:
        df_feat[f'lag_{lag}'] = df_feat[target_col].shift(lag)
        
    # 2. 构造滚动统计特征 (Rolling Features)
    # 捕捉局部趋势和波动：相当于 MA 模型思想的扩展
    df_feat[f'rolling_mean_{window}'] = df_feat[target_col].shift(1).rolling(window=window).mean()
    df_feat[f'rolling_std_{window}'] = df_feat[target_col].shift(1).rolling(window=window).std()
    
    # 3. 时间特征
    df_feat['month'] = df_feat.index.month
    df_feat['day_of_week'] = df_feat.index.dayofweek
    
    # 去除因shift产生的NaN
    return df_feat.dropna()

# 假设 df 是以时间为索引的 DataFrame
# df_processed = create_time_series_features(df, 'sales', lags=[1, 7, 14, 30], window=7)
```

### 3.3 效果与对比

与Transformer 相比，XGBoost 在时序任务上的优势在于：
1.  **训练速度极快**：无需像 LSTM/ViT 那样进行 BPTT。
2.  **对缺失值不敏感**：XGBoost 自带稀疏感知算法（Sparsity Aware Split Finding），自动学习缺失值的默认分裂方向。
3.  **可解释性强**：可以通过 Feature Importance 明确知道是“上周销量”还是“促销活动”主导了预测。

但缺点在于无法捕捉超长期的依赖（Long-term dependencies）以及无法外推（Extrapolation）到未见过的数值范围。

## 总结

本周重新审视了 XGBoost 这一机器学习界的基本算法，通过手推公式和模拟代码，重新尝试理解传统机器学习算法在时序应用上的独特优越之处。在  接触多模态风控数据后，发现对于数值型和类别型密集的表格数据，Transformer等深度模型往往需要极其复杂的 Embedding 设计才能匹敌 XGBoost 的简单暴力。在后续的学习实验中，我将进一步尝试深度学习 处理非结构化数据 + XGBoost 处理结构化数据 的两阶段融合策略作为一种可以考虑的实验方案。
