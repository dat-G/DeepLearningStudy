[TOC]

# Week 6: 深度学习补遗：半监督学习

## 摘要

本周继续跟进李宏毅老师的课程进行学习，本周主要对半监督学习的训练方式进行认识和学习，了解其背后的思想以及数学依据。

## Abstract

In this week's learning, I continued to follow up with Professor Li Hongyi's course, mainly to understand and learn about the training method of semi supervised learning, and to understand its underlying ideas and mathematical basis.

## 1. 监督学习和半监督学习

$$
\text{监督学习}:\underset{x^r为图像，\hat{y^r}为类别标签}{\{(x^r,\hat{y^r})\}^R_{r=1}} \\
\text{半监督学习}:\underset{x^u是一个未标注数据的集合，通常U>>R}{\{(x^r,\hat{y^r})\}^R_{r=1}, \{x^u\}^{R+U}_{u=R}}
$$

- Transductive Learning（直推学习）：未标注数据是测试数据。
- Inductive Learning（归纳学习）：未标注数据不是测试数据。

为什么选用半监督学习？因为**无标注数据收集难度较低**，但“带标注”数据**非常昂贵**（难以获取）。

为什么半监督学习有用？因为**未标注数据的分布会告诉我们一些信息**。

## 2. 监督学习和半监督学习的的生成式模型

给出带标注的训练样本$x^r\in C_1,C_2$

- 寻找最近似的先验概率$P(C_i)$和条件概率$P(x|C_i)$
- $P(x|C_i)$是一个由参数$\mu^i$和$\Sigma$控制的高斯分布

$$
P(\left.C_1\right|x) = \frac{P(\left.x\right|C_1)P(C_1)}{P(\left.x\right|C_1)P(C_1)+P(\left.x\right|C_2)P(C_2)}
$$

对于监督学习而言，训练会根据训练集给出一个明确的决策边界。

而对于半监督学习而言，未标注数据$x^u$会帮助重新估计$P(C_1)$、$P(C_2)$、$\mu^1$、$\mu^2$、$\Sigma$。

1. 初始化参数：$\theta=\{P(C_1), P(C_2), \mu^1,\mu^2,\Sigma\}$

2. 计算未标注数据的后验概率$P_\theta(\left.C_1\right|x^u)$，取决于模型$\theta$。

3. 更新模型，$N$为样本总数，$N_1$为属于$C_1$类别的样本数：
   $$
   P(C_1)=\frac{N_1+\sum_{x^u}P(\left.C_1\right|x^u)}{N}
   $$

   $$
   \mu^1=\frac{1}{N_1}\sum_{x^r\in C_1}x^r+\frac{1}{\sum_{x^u}P(\left.C_1\right|x^u)}\sum_{x^u}P(\left.C_1\right|x^u)x^u \\
   \dots
   $$
   回到第二步，重新计算，更新参数。

标注数据的极大似然估计为$\log L(\theta)=\sum_{x^r}\log P_\theta(x^r,\hat{y_r})$，有封闭解$P_\theta(x^r,\hat{y^r})=P_\theta(\left.x^r\right|\hat{y^r})P(\hat{y^r})$。

同时使用标注数据的极大似然估计为$\log L(\theta)=\sum_{x^r}\log P_\theta(x^r)+\sum_{x^u}\log P_\theta(x^u)$，可以迭代求解$P_\theta(x^u)=P_\theta(x^u|C_1)P(C_1)+P_\theta(x^u|C_2)P(C_2)$。其中的$x$可以来自于$C_1$或者$C_2$类别。

## 3. 自训练

给出带标注数据集${\{(x^r,\hat{y^r})\}^R_{r=1}}$，无标注数据集$\{x^u\}^{R+U}_{u=l}$。

重复过程：

- 从带标注数据集训练模型$f^*$（独立于模型）
- 将模型$f^*$应用到未标注数据集
  - 包含$\{(x^u,y^u)\}^{R+U}_{u=l}$（伪标签）
- 从未标注数据集中拿出一部分数据，并将他们加入带标注数据集中。（不同的伪标签有不同的置信度，可以根据一定的置信度规则筛选）

回归问题不能使用，因为预测出的输出本身已经是模型的预测结果，在反向传播时的损失就会是一个很小的数字，实际不会对模型造成影响，因此自训练对回归问题无效。

**自训练采用硬标签（Hard-Label）**，将置信度高的无标注数据的预测结果直接变为训练集进行训练。

**生成式模型采用软标签（Soft-Label）**，将无标注模型进行条件概率分布的预测，一部分归于Class 1，一部分归于Class 2，以此类推。

回归问题一定要采用软标签，而分类问题一定要采用硬标签，因为一个基于条件概率分布的预测不会改变原有分布，实际上不会起到改变模型的效果。

## 4. 熵正则

模型对数据的预测，应该尽量集中在一个类别上，这样区分效果会比较好，如何衡量预测结果的分布情况呢，就需要用到熵。
$$
E(y^u)=-\sum_{m=1}^5y^u_m\ln(y^u_m)
$$


熵越小代表分布越集中，这样子，半监督学习的目标就可以比较明确。即在带标注数据上，模型损失应该尽量的少；而在无标注数据上，熵应该尽量的小，使概率分布更加集中。因此目标函数就可以得到优化为：
$$
\begin{aligned}
L &= \underset{\text{Labelled Data}}{\sum_{x^r}C(y^r,\hat{y^r})}\\
&+\lambda\underset{\text{Unlabelled Data}}{\sum_{x^u}E(y^u)}
\end{aligned}
$$
加上熵正则的惩罚方式，就可以更好的避免模型进入过拟合。

## 5. 聚类方法

可以用聚类方法对无标签数据进行半监督学习，当数据量足够大的时候，在一个稠密区间中，可以近似的任务其中样本属于一个类别，步骤可以描述为：

1. 对数据进行聚类
2. 利用其中的带标签数据确定该聚类的类别，从而给无标签数据加上类别
3. 用新加上标签的有监督数据训练模型

聚类方法的关键在于选择合适的聚类手段，使一个聚类中的数据都属于一个类别，实际上Embedding的步骤并不简单，而且非常重要，对聚类效果产生很大的影响。

## 6. 基于图的方法

可以用图的方式进行数据关系的连接和建立，数据之间的关系，比如：网页链接间的跳转、商品点击的前后顺序、论文引用关系等。建立图和边，边可以有权重。高斯径向函数：
$$
d(x^i,y^i)=e^{(-\gamma||x^i+x^j||^2)}
$$
定义图中标签的平滑程度为：
$$
\begin{aligned}
S&=\frac{1}{2}\sum_{i,j}w_{i,j}(y^i-y^j)^2=\bold{y}^TL\bold{y} \\
\bold{y}&=[\dots y^i\dots y^j\dots]^T \text{(R+U)-dim vector} \\
\bold{L}&: \text{(R+U)}\times\text{(R+U) matrix}
\end{aligned}
$$
$L$是图的拉普拉斯矩阵，可以表示为$L=D-A$，其中$D$是图的度矩阵，$A$是图的邻接矩阵。

因此就可以改动损失函数为：
$$
\begin{aligned}
L &= \underset{\text{Labelled Data}}{\sum_{x^r}C(y^r,\hat{y^r})}\\
&+\lambda\underset{\text{Smoothness}}{S}
\end{aligned}
$$
用上述方法，就可以对输出进行图上的平滑。因此，利用图的方法，比如连通图或者强联通分量的方法，就可以对图进行聚类。但使用这种方法时，需要足够大的数据量以及足够全面的边连接关系，否则会造成节点之间的断裂。

## 总结

本周主要研究了半监督学习的方法，与监督学习的方法进行了对比，分辨了其底层逻辑的区别以及适用范围。同时，自训练与生成式模型也非常容易混淆，本周对其主要区别，硬标签与软标签进行了区分与理解。最后，学习了两种分别为基于图和基于聚类的自学习方法，同时还学习了熵正则以及平滑正则的正则化方法，对自学习有了相对全面的基本认识。
