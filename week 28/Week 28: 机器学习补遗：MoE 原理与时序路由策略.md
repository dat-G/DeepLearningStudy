[TOC]

# Week 28: 机器学习补遗：MoE 原理与时序路由策略

## 摘要

本周重点探讨了混合专家模型 (Mixture of Experts, MoE)。MoE是当前LLM中的研究热点，其思想与集成学习一脉相承，但以端到端可微的方式在深度神经网络中实现。

## Abstract

This week's focus has been on the Mixture of Experts (MoE) model. MoE represents a current research hotspot within large language models (LLMs), drawing inspiration from ensemble learning while being implemented in a fully differentiable manner within deep neural networks.

## 1. MoE（Mixture of Experts）

### 1.1 单一模型的局限性
在标准 Transformer 中，FeedForward Network (FFN) 对所有的输入 Token 使用同一组参数 $W$。
$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$
为了应对各种复杂情况，我们不得不把一个需要应对全部情况的模型做得非常巨大（增加参数量），但这导致了推理成本的飙升。

### 1.2 MoE 的核心思想
MoE 引入了“分诊”和“专科”的概念。它包含两个核心组件：
1.  **专家网络 (Experts, $E_i$)**：一组独立的神经网络（通常是 MLP），每个专家专注于处理输入空间的一个特定子区域。
2.  **门控网络 (Gating Network / Router, $G$)**：一个轻量级的分类器，根据输入 $x$ 的特征，决定派哪个（或哪几个）专家上场。

其通用数学形式为：
$$
y = \sum_{i=1}^N G(x)_i \cdot E_i(x)
$$
这与XGBoost中将样本划分到不同叶子节点的思想异曲同工，区别在于 MoE 的划分是软性 (Soft)且动态 (Dynamic)的，并且全程通过梯度下降联合训练。

## 2. Sparse MoE

### 2.1 Dense MoE 的计算瓶颈
在原始的 MoE 定义中，门控网络 $G(x)$ 输出的是一个针对所有 $N$ 个专家的概率分布（Sum = 1）。这意味着对于每一个输入，**所有的**专家都要进行前向传播计算。
当专家数量扩展到成百上千时，计算量将线性增长，这违背了我们想通过增加参数来提升模型容量的初衷。

### 2.2 Sparse MoE与Top-K Gating
为了解决这个问题，Sparse MoE引入了稀疏性约束。核心思想是条件计算 (Conditional Computation)。即对于每个样本，我们只激活一小部分专家。

这是通过Top-K Gating实现的。假设我们有 $N$ 个专家，我们只选取门控分数最高的 $K$ 个（通常 $K=1$ 或 $2$）：

$$
\text{TopK}(v, k)_i = \begin{cases} 
v_i & \text{if } v_i \text{ is in the top } k \text{ elements of } v \\
-\infty & \text{otherwise}
\end{cases}
$$

具体的门控输出计算如下：
1.  计算原始路由分数：$h(x) = x \cdot W_g$
2.  加入高斯噪声（可选，用于负载均衡）：$h'(x) = h(x) + \text{StandardNormal}() \cdot \text{Softplus}(x \cdot W_{noise})$
3.  应用 Softmax 并保留 Top-K：
    $$
    G(x) = \text{Softmax}(\text{TopK}(h'(x), k))
    $$

**意义**：通过这种方式，我们可以拥有一个参数量高达数千亿的“超大模型”，但每次推理只调用其中几十亿的参数。实现了“大模型的容量，小模型的速度”。

## 3. 路由粒度 (Routing Level)

 路由粒度 是将 MoE 应用于时序预测时最关键，也是最容易被忽视的环节。

### 3.1 Token Level MoE (Token 级路由)
在 NLP 领域（如 GPT-4 或 Switch Transformer），路由是针对每个 Token 独立进行的。
*   **过程**：对于时间序列 $X = \{x_1, x_2, ..., x_T\}$，Router 可能会把 $x_1$ 分给专家 A，把 $x_2$ 分给专家 B。
*   **时序领域的弊端**：
    *   **破坏连续性**：时间序列具有极强的自相关性（Autocorrelation）。如果相邻时间步由完全不同的专家网络处理，输出结果可能会出现不自然的跳变 (Jumps)。
    *   **缺乏上下文**：单个时间步 $x_t$ 的数值可能本身不包含足够的信息来判断当前的“市场状态”（Regime）。例如，价格为 100 时，可能是上涨趋势中的 100，也可能是下跌趋势中的 100。仅凭 $x_t$ 很难区分。

### 3.2 Pooling Level MoE (池化级/序列级路由)
针对上述问题，**Pooling Level MoE** 提出了一种更符合时序特性的策略：**以时间窗口（Patch 或 Whole Series）为单位进行路由**。

*   **核心逻辑**：
    1.  **Pooling**：首先提取一段时间内的全局特征。
        $$
        v_{global} = \text{AveragePool}(x_1, ..., x_T) \quad \text{或} \quad v_{global} = \text{AttentionPool}(X)
        $$
    2.  **Routing**：基于这个全局特征 $v_{global}$ 来决定使用哪个专家。
        $$
        \text{Selected Expert} = \text{TopK}(G(v_{global}))
        $$
    3.  **执行 (Execution)**：被选中的专家处理**整个**时间窗口的数据。

*   **学术解释**：
    这种做法实际上是对非平稳性 (Non-stationarity) 的一种显式建模。我们假设时间序列是由若干个隐含的分布状态 (Regimes)组成的（例如：平稳震荡期、剧烈波动期）。
    
    *   Pooling 操作旨在识别当前窗口属于哪个 Regime。
    *   不同的 Experts 实际上习得了不同 Regime 下的动力学方程。
    *   Routing 保证了在同一个 Regime 内部，预测逻辑的一致性和平滑性。

## 总结

本周比较详细的研究了MoE的数学原理以及从Dense MoE到Sparse MoE的演进逻辑，并针对时间序列数据的非平稳性和时序连续性，深入剖析了Pooling Level MoE相较于传统 Token Level 路由的必要性。后续可能考虑对MoE继续进行深入研究，MoE作为当下的研究热点，可以作为一个策略或者模块比较容易的嵌入现有的网络当中，值得继续研究。
