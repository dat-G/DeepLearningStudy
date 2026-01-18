@[toc]
# Week 26: 深度学习补遗：LSTM 原理与代码复现

## 摘要

本周对 LSTM模型利用PyTorch进行了复现，将其实现与数学形式进行了逐条对应，清晰、重点地理解门控机制与细胞状态与其在时序场景下的应用。

## Abstract

This week, I replicated the LSTM model using PyTorch, aligning its implementation with the mathematical formulation step by step. This enabled a clear and focused understanding of the gating mechanisms, cellular states, and their application within temporal contexts.

## 1. LSTM数学回顾

LSTM 是为了解决标准循环神经网络（RNN）在处理长序列时出现的梯度消失（Gradient Vanishing）问题而设计的。标准 RNN 像是一个患有“短期健忘症”的人，很难记住很久之前的输入对当前的影响。

LSTM 通过引入一个核心概念——细胞状态（Cell State），以及一套门控机制（Gating Mechanism）来解决这个问题。在这个细胞状态上，信息只会发生线性的加减变化，这使得梯度在反向传播时能够顺畅地流回早期的时刻，从而保留长距离的依赖关系。

LSTM 通过三个“门”来控制向传送带上添加什么信息，或者从传送带上擦除什么信息。这些门本质上是神经网络层（通常使用 Sigmoid 激活函数），输出 0 到 1 之间的数值（0 代表完全关闭/遗忘，1 代表完全开启/保留）。

*   **遗忘门 (Forget Gate)**：
    *   **功能**：决定我们要从细胞状态中**丢弃**什么信息。
    *   **例子**：在文本预测中，如果模型看到了一个新的主语（如“她”），它可能需要“遗忘”之前的主语（如“他”），以便正确匹配后续的代词。

*   **输入门 (Input Gate)**：
    *   **功能**：决定我们要向细胞状态中**存储**什么新信息。它包含两部分：一部分决定更新哪些值（Sigmoid），另一部分生成新的候选值（Tanh）。
    *   **例子**：将新的主语信息写入记忆。

*   **输出门 (Output Gate)**：
    *   **功能**：基于当前的细胞状态，决定我们要**输出**什么值给下一个时间步或作为最终预测。
    *   **例子**：既然知道主语是单数，模型决定输出一个单数形式的动词。

单层 LSTM 在时间步 $t$ 的核心运算为：
$$
\begin{aligned}
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i)\\
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f)\\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o)\\
	ilde{c}_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c)\\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t\\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

其中 $x_t\in\mathbb{R}^{d_{in}}$，$h_t,c_t\in\mathbb{R}^{d_h}$；$\sigma$ 为 Sigmoid，$\odot$ 表示逐元素乘积。

在 PyTorch 中，`nn.LSTM` 将上述门对应的线性变换合并存储（例如 `weight_ih_l0` 同时包含 $W_i,W_f,W_o,W_c$ 的系数），以一次矩阵乘法并行计算来提升性能。

## 2. LSTM简要代码实现

### 2.1 模型

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
```

- `input_size` 对应 $d_{in}$，`hidden_size` 对应 $d_h$。
- `batch_first=True` 表示输入张量格式为 `(batch, seq_len, feat)`，这会影响后续按时间步索引的写法。
- 当 `num_layers>1` 时，`nn.LSTM` 会在层间自动应用 dropout；此外外部 `self.dropout` 用于对时间步输出做一次额外的正则化。
- `fc` 将最后时间步的隐藏表示 $h_T$ 映射到预测空间（例如回归标量）。

### 2.2 隐藏与细胞状态初始化

```python
def init_hidden(self, batch_size: int):
    h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
    c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
    return h0, c0
```

- 返回形状 `(num_layers, batch_size, hidden_size)` 的零张量，用作 $(h_0, c_0)$。若使用双向 LSTM，此处需乘以 2。

### 2.3 前向传播

```python
def forward(self, x, hidden=None):
    batch_size = x.size(0)
    if hidden is None:
        hidden = self.init_hidden(batch_size)
    lstm_out, hidden = self.lstm(x, hidden)
    lstm_out = self.dropout(lstm_out)
    out = self.fc(lstm_out[:, -1, :])
    return out, hidden
```

- 输入 `x` 形状为 `(batch, seq_len, input_size)`。
- `self.lstm(x, hidden)` 返回：
  - `lstm_out`，形状 `(batch, seq_len, hidden_size)`，包含每个时间步 $h_t$；
  - `hidden=(h_n,c_n)`，其中 `h_n` 形状 `(num_layers, batch, hidden_size)`。
- `lstm_out[:, -1, :]` 取最后时间步的 $h_T$（顶层输出的最后时刻），形状 `(batch, hidden_size)`，再通过 `fc` 得到最终预测。

### 2.4 PyTorch 中门的向量化实现

- PyTorch 使用合并矩阵来并行计算四个门：这意味着底层先做一次 `W x + b_ih` 与 `U h + b_hh`，再将所得大向量切分成四份对应 `i,f,o,g`，分别使用激活函数进行激活。该实现能显著降低矩阵乘法次数并提升效率。

## 3. 总结与使用场景

在 2017 年 Transformer 架构（Attention 机制）横空出世之前，LSTM 曾是序列建模（NLP、语音、时序）的绝对王者。而在 2025 年的今天，它不再是唯一的 SOTA，但依然是非常棒的轻量级选择。

### 3.1 LSTM的显著劣势

*   **并行计算能力的缺失**：LSTM 是串行的（必须等 $t-1$ 算完才能算 $t$），这导致它无法像 Transformer 那样充分利用 GPU 的并行计算能力，在大规模数据训练上效率较低。
*   **超长序列的瓶颈**：虽然 LSTM 解决了梯度消失，但在处理极长序列（例如长度超过 1000-2000 的时间步）时，其记忆能力依然不如基于注意力机制的模型（如 Informer, Autoformer 等）。

### 3.2 LSTM的优势

尽管有 Transformer 和 TCN的挑战，LSTM 在时序预测领域依然非常活跃，原因如下：

*   数据效率（Data Efficiency）：Transformer 类模型通常需要海量数据才能训练好（Data Hungry）。而在许多实际业务场景中，数据量可能只有几千条，此时 LSTM 往往能比 Transformer 取得更好的效果，且不易过拟合。
*   归纳偏置（Inductive Bias）：RNN 结构的递归特性天然契合时间序列的“因果性”（当前时刻由过去时刻决定）。
*   部署成本：LSTM 模型通常参数量较小，推理速度快，非常适合部署在边缘设备或资源受限的服务器上。

## 总结

本周阅读了一些时序预测领域的工作，发现LSTM在时序预测领域仍然相当活跃，在本周对LSTM进行了数学原理的复习、代码的简单复现和其应用场景的分析。LSTM在一些数据量较小的场景仍然非常强势，而其精简的设计结构也造就了非常强大的改进潜力，可以往其中添加比较符合场景的计算模块也不会增加过多收敛难度，仍然在时序场景具有比较大的意义。