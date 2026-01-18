[TOC]

# Week 31: 深度学习补遗：Mamba

## 摘要

本周的研究了2024年的热门深度学习架构Mamba。从底层的微分方程离散化 (Discretization)入手，理解了 Mamba 的核心创新——选择性扫描 (Selective Scan)，解释了模型是如何通过动态调整时间步长 $\Delta$ 来实现类似于 LSTM 门控的“遗忘”与“记忆”机制。

## Abstract

This week's research examined Mamba, a prominent deep learning architecture for 2024. Beginning with the underlying discretisation of differential equations, we explored Mamba's core innovation—Selective Scan—and elucidated how the model achieves LSTM-like gating mechanisms for "forgetting" and "remembering" by dynamically adjusting the time step size $\Delta$.

## 1. 连续微分到离散递归

### 1.1 连续系统的物理意义
SSM 的起点是描述物理系统随时间变化的微分方程。
$$
h'(t) = \mathbf{A}h(t) + \mathbf{B}x(t)
$$
$h(t)$ 是系统的“当前状态”（例如弹簧的位置和速度）。$\mathbf{A}$ 矩阵描述系统在没有外力时的自然演变（例如弹簧的阻尼衰减）。$\mathbf{B}x(t)$ 是外力输入对状态的影响。

### 1.2 零阶保持 (ZOH) 
在深度学习中，数据不是连续的，而是离散的采样点 $x_0, x_1, \dots, x_k$。我们需要将上述微分方程转化为递归式：$h_t = \overline{\mathbf{A}} h_{t-1} + \overline{\mathbf{B}} x_t$。

假设在时间区间 $[t, t+\Delta]$ 内，输入 $x(t)$ 保持恒定值 $x_k$（即“零阶保持”）。

对微分方程两边积分，解得 $t+\Delta$ 时刻的状态：
$$
h(t+\Delta) = e^{\Delta \mathbf{A}} h(t) + \int_{t}^{t+\Delta} e^{(t+\Delta-\tau)\mathbf{A}} \mathbf{B} x(\tau) d\tau
$$

由于假设 $x(\tau)$ 在该区间是常数，可以提到积分号外面。积分部分 $\int_{0}^{\Delta} e^{u\mathbf{A}} du = \mathbf{A}^{-1}(e^{\Delta \mathbf{A}} - \mathbf{I})$。

由此得到具体的离散化参数公式：
1.  **状态转移矩阵 $\overline{\mathbf{A}}$**：
    $$
    \overline{\mathbf{A}} = \exp(\Delta \cdot \mathbf{A})
    $$
    
    *   **具体含义**：$\Delta$ 越大，$\overline{\mathbf{A}}$ 变化越剧烈。如果 $\mathbf{A}$ 是负的（通常为了稳定性设为对角负矩阵），$\exp(\Delta \mathbf{A})$ 就会趋近于 0。
    *   时间步长 $\Delta$ 越大，系统遗忘历史信息越快。
    
2.  **输入投影矩阵 $\overline{\mathbf{B}}$**：
    $$
    \overline{\mathbf{B}} = (\Delta \mathbf{A})^{-1} (\exp(\Delta \mathbf{A}) - \mathbf{I}) \cdot \Delta \mathbf{B}
    $$
    
    *   **近似**：在 $\Delta$ 很小时，泰勒展开第一项主导，可近似为 $\overline{\mathbf{B}} \approx \Delta \cdot \mathbf{B}$。
    *   时间步长 $\Delta$ 越大，当前输入 $x_t$ 对状态的影响权重越大。

## 2. 选择性扫描 (Selective Scan) 

### 2.1 传统LTI系统
在 Mamba 之前的 S4 模型中，$\Delta, \mathbf{A}, \mathbf{B}$ 都是**静态参数**（训练完就固定了）。
这相当于模型用**同一套滤波器**处理所有数据。

假设输入序列是 `[有用信息, 噪音, 噪音, 有用信息]`。LTI 系统无法在遇到“噪音”时主动切断记忆，也无法在遇到“有用信息”时以此为重。

### 2.2 Mamba 的动态门控机制
Mamba 将参数变成了输入的函数：
$$
(\Delta_t, \mathbf{B}_t, \mathbf{C}_t) = \text{Linear}(x_t)
$$


这意味着对于序列中的每一个 Token $x_t$，模型都会生成一套**独一无二**的离散化参数。

1.  需要忽略的噪声
    *   模型检测到 $x_t$ 是噪声。
    *   预测出的 $\Delta_t$ **变大**（例如从 0.1 变为 10）。
    *   结果：$\overline{\mathbf{A}}_t = \exp(-10) \approx 0$。
    *   **效果**：$h_t = 0 \cdot h_{t-1} + \dots$，历史状态被清空/遗忘，噪声没有被长期记忆。

2.  需要保留的关键信号
    *   模型检测到 $x_t$ 是关键信号。
    *   预测出的 $\Delta_t$ **变小**，且 $\mathbf{B}_t$ 变大。
    *   结果：$\overline{\mathbf{A}}_t \approx 1$，$\overline{\mathbf{B}}_t$ 很大。
    *   **效果**：历史记忆无损保留，当前输入被强力写入状态。

这就是 Mamba 被称为“具有选择性”的具体原因——它通过动态调整时间刻度 $\Delta$，实现了类似于 LSTM 中“遗忘门”和“输入门”的功能，但计算效率更高。

## 3. 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlockSpecific(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model  # D: 输入维度 (例如 512)
        self.d_state = d_state  # N: 状态维度 (例如 16)
        
        # 定义 A, D 参数
        self.A_log = nn.Parameter(torch.log(torch.randn(d_model, d_state).abs()))
        self.D = nn.Parameter(torch.ones(d_model))
        
        # 核心投影层
        self.in_proj = nn.Linear(d_model, d_model * 2)
        # x_proj 负责从输入生成 B, C, Delta
        # 输出维度是 N + N + 1 (B的状态, C的状态, Delta的标量)
        self.x_proj = nn.Linear(d_model, (d_state * 2) + 1) 
        self.dt_proj = nn.Linear(1, d_model)

    def forward(self, x):
        # x Shape: [Batch, Seq_Len, D]
        B, L, D = x.shape
        N = self.d_state
        
        # 1. 扩展输入维度
        x_and_res = self.in_proj(x)  # [B, L, 2*D]
        x_in, res = x_and_res.chunk(2, dim=-1) # x_in: [B, L, D], res: [B, L, D]
        
        # ----------------------------------------------------
        # 2. 动态参数生成 (Discretization & Selection)
        # ----------------------------------------------------
        # 这一步是 Mamba 区别于 S4 的关键：参数随 t 变化
        ssm_params = self.x_proj(x_in) # [B, L, 2*N + 1]
        
        # 切分出 B_t, C_t, dt_t
        B_t = ssm_params[:, :, :N]      # [B, L, N]
        C_t = ssm_params[:, :, N:2*N]   # [B, L, N]
        dt_t = ssm_params[:, :, 2*N:]   # [B, L, 1]
        
        # 广播 Delta: 将标量 Delta 投影回特征维度 D
        dt_t = F.softplus(self.dt_proj(dt_t)) # [B, L, D]
        
        # 计算离散化的 A_bar (Decay Rate)
        # A 是 (D, N), dt_t 是 (B, L, D) -> 广播计算
        # exp(Delta * A) 决定了记忆衰减的速率
        A = -torch.exp(self.A_log) 
        dA = torch.exp(torch.einsum('bld,dn->bldn', dt_t, A)) # [B, L, D, N]
        
        # 计算离散化的 B_bar (Input Weight)
        # B_bar = Delta * B
        dB = torch.einsum('bld,bln->bldn', dt_t, B_t) # [B, L, D, N]
        
        # ----------------------------------------------------
        # 3. 状态空间扫描 (SSM Scan)
        # ----------------------------------------------------
        # 初始化隐状态 h: [B, D, N]
        # 注意：这里的 h 是 latent state，相比 Transformer 的 KV Cache 极小
        h = torch.zeros(B, D, N, device=x.device)
        ys = []
        
        # 串行扫描演示 (实际 CUDA kernel 会使用并行前缀和优化)
        for t in range(L):
            # h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
            # 具体操作：
            # 1. 衰减历史: h * dA[:, t] (按元素乘，不同通道衰减率不同)
            # 2. 写入新值: x_in[:, t] * dB[:, t]
            h = dA[:, t] * h + dB[:, t] * x_in[:, t].unsqueeze(-1)
            
            # y_t = C_t * h_t
            # 将隐状态 N 投影回输出维度 D
            y_t = torch.einsum('bdn,bn->bd', h, C_t[:, t])
            ys.append(y_t)
            
        y = torch.stack(ys, dim=1) # [B, L, D]
        
        # 残差连接与输出
        y = y + x_in * self.D
        return y * F.silu(res)
```

## 4. RNN、Transformer和Mamba的优劣

相比于 RNN (LSTM/GRU)，Mamba 克服了其门控机制虽动态但无法利用 GPU 并行训练的瓶颈。

Mamba 虽然在逻辑上保持了递归形式（$h_t = \overline{\mathbf{A}}_t h_{t-1} + \dots$），但由于采用了不包含 $\tanh$等非线性激活的线性递归，从而可以利用数学上的结合律引入并行前缀扫描算法，实现了像 Transformer 一样极速的并行训练。

而在与 Transformer 的对比中，Mamba 解决了 Attention 矩阵需存储 $L \times L$ 历史交互以及推理时 KV Cache 显存占用巨大的问题。Mamba 在推理时仅需维护一个大小为 $D \times N$ 的固定状态 $h_t$，这意味着无论序列长度是 1k 还是 100k，其推理所需的显存和计算量始终保持**恒定**。

## 总结

本周学习了Mamba，不同于 Transformer 的注意力机制，Mamba 是状态空间模型 (SSM)*的集大成者。通过具体推导了零阶保持 (ZOH) 技术的数学细节，我理解了选择性扫描 (Selective Scan)是如何实现类似门控的效果的。而代码复现部分，我更重点聚焦了动态参数生成的具体维度变化与数据流向。Mamba有其局限性，但其模块具有比较显著的创新性，可以考虑后续融合一部分模块在未来研究中。