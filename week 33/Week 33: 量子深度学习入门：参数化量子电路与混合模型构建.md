[TOC]

# Week 33: 量子深度学习入门：参数化量子电路与混合模型构建

## 摘要

本周初探了量子机器学习领域。利用GPU对量子电路进行了模拟。本周理解并构建参数化量子电路，通过将其视为一个可微的“量子层”嵌入到经典神经网络中，实现了经典-量子混合模型的端到端训练。

## Abstract

This week, I made initial forays into the field of quantum machine learning. Utilising GPUs, I simulated quantum circuits. I gained an understanding of and constructed parameterised quantum circuits, embedding them as differentiable 'quantum layers' within classical neural networks to achieve end-to-end training of classical-quantum hybrid models.

## 1. 理论基础：量子神经元

### 1.1 从比特到量子比特 (Qubit)
经典深度学习的基础是比特（0 或 1），而量子计算的基础是 Qubit。一个 Qubit 的状态 $|\psi\rangle$ 可以表示为基态 $|0\rangle$ 和 $|1\rangle$ 的线性叠加：
$$
|\psi\rangle = \alpha |0\rangle + \beta |1\rangle
$$
其中 $\alpha, \beta \in \mathbb{C}$ 且 $|\alpha|^2 + |\beta|^2 = 1$。这不仅仅是概率分布，而是复数概率幅，意味着量子态之间可以发生干涉 (Interference)——这是量子计算算力的核心来源。

### 1.2 参数化量子电路
在深度学习中，我们通过调整权重 $W$ 来拟合函数。在量子计算中，我们通过调整量子门 (Quantum Gates) 的旋转角度 $\theta$ 来演化量子态。

一个典型的 PQC 包含三个阶段：
1.  编码：将经典数据 $x$ 转化为量子态 $|\psi_x\rangle$（例如使用 Rotation Encoding）。
2.  演化 (Ansatz)：应用一系列带参数 $\theta$ 的旋转门（如 $R_x(\theta), R_y(\theta)$）和纠缠门（如 CNOT），将量子态变换为 $|\psi(\theta, x)\rangle$。这等价于经典网络中的前向传播。
3.  测量：对量子态进行测量，计算期望值 $\langle Z \rangle$，将量子信息坍缩回经典数值输出。

数学上，这个过程是：
$$
f(x; \theta) = \langle 0| U^\dagger(x) V^\dagger(\theta) \hat{O} V(\theta) U(x) |0\rangle
$$
这就构建了一个量子神经元。

## 2. 量子梯度下降

### 2.1 量子电路的训练？
要将量子电路嵌入 PyTorch，必须能够计算梯度 $\partial f / \partial \theta$。
对于常用的旋转门（如 $R_x(\theta) = e^{-i\theta X/2}$），我们使用参数平移规则 (Parameter-Shift Rule) 来计算解析梯度：

$$
\frac{\partial f}{\partial \theta} = \frac{f(\theta + \frac{\pi}{2}) - f(\theta - \frac{\pi}{2})}{2}
$$

这非常神奇：它意味着我们不需要深入量子态的内部（那通常是指数级复杂的），只需要在两个不同的参数点运行电路，就能精确算出梯度。这使得 PQC 可以无缝接入 Backpropagation 算法。

## 3. 构建经典-量子混合网络

使用了 `PennyLane` 库并配合 PyTorch 接口可以利用 GPU 加速模拟（模拟量子门本质上是矩阵乘法。

### 3.1 环境配置与电路定义

```python
import pennylane as qml
import torch
import torch.nn as nn

# 定义量子设备 (使用 default.qubit 模拟器)
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """
    inputs: 经典输入数据 (Batch, n_qubits)
    weights: 可训练参数
    """
    # 1. 编码层: 将经典数据映射到量子态 (Angle Encoding)
    # 类似于 input layer
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # 2. 变分层 (Ansatz): 类似于 hidden layers
    # BasicEntanglerLayers 包含了一层旋转门和一层纠缠门
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    
    # 3. 测量层: 输出每个 qubit 的 Pauli-Z 期望值
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
```

### 3.2 混合模型架构

我们将上述量子电路包装成一个 `QuantumLayer`，夹在两个经典 Linear 层之间，构建一个用于 MNIST 分类的混合模型。

```python
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 经典预处理层: 将 28x28 图片降维到 4 (对应 Qubit 数)
        self.clayer_1 = nn.Linear(28*28, n_qubits)
        
        # 量子层参数初始化
        # 2层结构，每层每个qubit有一个旋转参数
        weight_shapes = {"weights": (2, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # 经典后处理层: 将量子输出映射到 10 类
        self.clayer_2 = nn.Linear(n_qubits, 10)

    def forward(self, x):
        # x shape: (batch, 1, 28, 28)
        x = x.view(-1, 28*28)
        
        x = self.clayer_1(x)
        x = torch.tanh(x) # 将数据压缩到 [-1, 1] 或 [0, pi] 供量子编码
        
        # 进入量子层
        # 输入是经典的，内部演化是量子的，输出又是经典的
        x = self.qlayer(x)
        
        x = self.clayer_2(x)
        return x

# 之后可以像训练普通 CNN 一样使用 CrossEntropyLoss 和 SGD 训练此模型
```

## 4. 量子模拟的意义与瓶颈

### 4.1 GPU模拟的可行性
真正的量子计算机（QPU）目前噪音很大（NISQ 时代），且访问昂贵。
但在 GPU 上模拟量子电路，本质上是在进行大规模的复数矩阵乘法。

*   $N$ 个 Qubits 的状态向量大小是 $2^N$。
*   对于 $N < 30$，现代 GPU (如 A100) 可以极快地进行全状态向量模拟。
这让我们可以在没有量子计算机的情况下，验证量子算法的逻辑和梯度下降的可行性。

### 4.2 表达能力
研究表明，PQC 的表达能力与量子纠缠（Entanglement）密切相关。纠缠门（如 CNOT）让 Qubit 之间产生关联，这在数学上类似于经典网络中的非线性激活函数。没有纠缠的量子电路，仅仅是线性变换，表达能力有限。

### 4.3 贫瘠高原问题
这是 QML 领域的“梯度消失”问题。当量子电路过深或 Quibit 过多时，损失函数的梯度方差会指数级衰减至 0。这使得训练深层量子网络极其困难。这也解释了为什么目前的 QML 架构多采用 "浅层量子 + 深层经典" 的三明治结构。

## 总结

本周对参数化量子神经网络进行了初步的了解，这周的学习让我明白，QML 不是要取代经典深度学习，而是作为一种高性能的Kernel或特征提取器，与经典网络协同工作。接下来的学习将进一步了解如何设计更好的 Ansatz，以捕捉数据中经典方法难以察觉的相关性。