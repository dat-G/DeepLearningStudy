[TOC]

# Week 32: 深度学习补遗：Agent的认知架构、记忆系统与高阶规划

## 摘要

本周了解了Agent 的认知架构，在理解基础的 ReAct范式之上，进一步深入研究了支持长程任务的记忆系统与高阶规划。本周重点探讨了如何利用外部向量存储扩展 LLM 的“海马体”，以及如何通过思维树和自我反思机制，让 Agent 具备从错误中学习和处理非线性复杂问题的能力。

## Abstract

This week, we delved into the cognitive architecture of Agents, building upon our understanding of the fundamental ReAct paradigm to further investigate memory systems and high-level planning that support long-term tasks. The focus of this week was on exploring how to expand the "hippocampus" of LLMs using external vector storage, as well as how to equip Agents with the ability to learn from errors and handle nonlinear complex problems through thought trees and self-reflection mechanisms.

## 1. ReAct范式与认知方程

### 1.1 交互式决策的数学形式

传统的 LLM 是被动的，其目标函数是最大化静态数据的似然概率。而 Agent 是主动的，它可以被形式化为一个在环境中互动的决策过程。
一个通用的 Agent 系统由四大支柱构成：
$$
\text{Agent} = \text{LLM (Brain)} + \text{Memory} + \text{Planning} + \text{Tools}
$$

在时间步 $t$，Agent 的行为 $a_t$ 取决于当前的观察 $o_t$ 和累积的记忆 $h_t$：
$$
a_t \sim \pi_\theta(a_t | o_t, h_t)
$$

### 1.2 ReAct

ReAct (Reason + Act)范式是 Agent 的原子单元。它打破了“输入-输出”的黑盒，强制模型生成显式的推理轨迹 (Reasoning Trace)。

其循环本质是：
$$
\text{Thought}_t \rightarrow \text{Action}_t \rightarrow \text{Observation}_t \rightarrow \text{Thought}_{t+1}
$$

这种机制在数学上等价于利用 LLM 的上下文窗口作为工作记忆 (Working Memory)，将环境的反馈 $obs$ 重新注入模型，从而修正先验的幻觉。

## 2.记忆与规划 (Memory & Planning)

单纯的 ReAct 循环在面对简单任务时表现良好，但在处理需要多步推理或长期依赖的复杂任务时，容易陷入循环或迷失方向。因此，引入更高级的记忆和规划模块至关重要。

### 2.1 记忆系统

人类的记忆分为短期记忆（工作记忆）和长期记忆。Agent 的架构设计借鉴了神经科学的这一原理：

1.  短期记忆 (Short-term Memory)：
    *   实现：直接映射为 LLM 的 Context Window。
    *   局限：受限于 Transformer 的序列长度（尽管 Mamba 等架构在缓解此问题，但 Prompt 越长，注意力越分散）。
    *   作用：存储当前的推理轨迹和临时的变量。

2.  长期记忆 (Long-term Memory)：
    * 实现：外部 向量数据库 (Vector Database)。
    
    * 机制：通过快速检索 (Retrieval) 机制，$R(q) = \text{TopK}(\text{Sim}(E(q), E(D)))$，将相关的历史经验提取到 Context 中。
    

这赋予了 Agent 跨越时间周期的学习能力。例如，Agent 可以记住用户偏好的编码风格，或者在遇到类似错误时调用之前的解决方案。

### 2.2 高阶规划

ReAct 默认是线性的贪婪解码。为了解决复杂问题（如“写一个贪吃蛇游戏”），Agent 需要具备规划 (Planning) 能力。

1. 任务分解 (Decomposition)
  利用 Chain of Thought (CoT)，将宏大目标 $G$ 拆解为子目标序列 $\{g_1, g_2, ..., g_n\}$。
  $$
  \text{Plan}: G \rightarrow g_1 \rightarrow g_2 \dots
  $$
  Agent 依次执行 $g_i$，每一步只关注当前的子任务。

2. 思维树 (Tree of Thoughts, ToT)
  这是对 CoT 的推广。CoT 是单路径推理，而 ToT 允许 Agent 在思维空间中进行搜索（如 BFS 或 DFS）。

  *   生成 (Generation)：针对当前状态 $s$，生成 $k$ 个可能的下一步思维 $z^{(1)}, ..., z^{(k)}$。
  *   评估 (Evaluation)：自我评估每个思维的可行性 $V(s, z)$。
  *   搜索 (Search)：选择最优路径，甚至支持回溯 (Backtracking)。

  $$
  z^* = \mathop{\arg\max}_{z \in \mathcal{Z}} V(p_\theta, z | x)
  $$
  这种机制让 Agent 具备了“三思而后行”的能力，在写代码或数学证明时尤为有效。

3. 自我反思 (Self-Reflection)
  Reflexion 框架提出了一种能够自我修正的机制。当 Agent 任务失败时，它不是立即停止，而是生成一个口头反馈 (Verbal Feedback)，存入长期记忆。
  $$
  \text{Trace}_{new} = \text{Trace}_{old} + \text{Critique}(\text{Trace}_{old})
  $$
  在下一次尝试时，Agent 会读取这个“自我批评”，避免重蹈覆辙。

## 3. 深度思考

在复现和研究这些 Agent 架构时，我深刻体会到了 Daniel Kahneman 在《思考，快与慢》中提出的认知双系统理论在 AI 中的投射：

1.  System 1 (快思考)：
    *   对应 LLM 的直觉输出。直接 Prompting，反应快，但容易产生幻觉，逻辑跳跃。
    *   传统的 Zero-shot / Few-shot 属于此类。

2.  System 2 (慢思考)：
    *   对应 Agent 的规划架构。通过 ReAct 循环、ToT 搜索、Reflexion 反思。
    *   虽然牺牲了推理速度（Latency 增加），但通过消耗更多的计算时间（Test-time Compute），换取了极高的逻辑准确性和任务完成率。

AI 的进步可能不仅仅来自于把模型做大，更来自于把认知架构设计得更精巧。Memory 解决了“遗忘”的问题，Planning 解决了“短视”的问题。

## 总结

本周开始学习一些Agent相关的基础理论知识，其本质上是一个运行在 LLM 之上的操作系统。后续除了继续学习深度学习外，尝试了解一些知识图谱、量子计算融合深度学习的相关知识，拓宽项目、比赛、论文所需的知识储备。