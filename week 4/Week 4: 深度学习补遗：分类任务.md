[TOC]

# Week 4: 深度学习补遗：分类任务

## 摘要

本周将继续学习李宏毅老师的课程。主要深入探讨模型训练中的两个核心议题：过拟合与分类任务。在本周的学习过程中，理解过拟合问题以及了解其解决方案。同时，对分类任务的数学原理，包括概率模型、极大似然估计以及后验概率等的数学原理进行了了解、阐述与推导。

## Abstract

This week, I continued to study Professor Hung-yi Lee's course. I mainly explored two core issues in model training: overfitting and classification tasks. In this week's learning process, I understood the overfitting problem and its solution. At the same time, I understood, explained and deduced the mathematical principles of classification tasks, including probability models, maximum likelihood estimation and posterior probability.

## 1. 过拟合

跟随着李宏毅老师的课程，在课程[ML讲座1：回归 - 案例研究](https://youtu.be/fegAeph9UaA?list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&t=2938)中，利用十个宝可梦进化后的CP值与进化前的CP值作为训练集，设定五个简单的函数模型进行训练，分别带有1到$n$次项。
$$
\begin{cases}
y=b + w\cdot x_{cp} \\
y=b + w_1\cdot x_{cp} + w_2\cdot x_{cp}^2 \\
y=b + w_1\cdot x_{cp} + w_2\cdot x_{cp}^2 + w_3\cdot x_{cp}^3 \\
y=b + w_1\cdot x_{cp} + w_2\cdot x_{cp}^2 + w_3\cdot x_{cp}^3 + w_4\cdot x_{cp}^4 \\
y=b + w_1\cdot x_{cp} + w_2\cdot x_{cp}^2 + w_3\cdot x_{cp}^3 + w_4\cdot x_{cp}^4 + w_5\cdot x_{cp}^5
\end{cases}
$$

训练结果如下：

|          | 1    | 2    | 3    | 4    | 5     |
| -------- | ---- | ---- | ---- | ---- | ----- |
| Training | 31.9 | 15.4 | 15.3 | 14.9 | 12.8  |
| Testing  | 35.0 | 18.4 | 18.1 | 28.2 | 232.1 |

可见随着模型复杂度提升，训练集上的$Loss$逐渐降低，但是测试集的$Loss$在模型加入四次项以后开始暴增，但训练集上的$Loss$仍在降低，这就是出现了**过拟合**。

在过拟合出现后，有多种思路可以考虑用于减低过拟合程度。

### 1.1 数据集优化

首先，可以增加数据量，提高数据集的多样化程度，从而提高模型的泛化能力。

### 1.2 模型结构优化

其次，可以考虑重新设计模型结构，使其能够更加恰当的适应应用场景。例如，对于宝可梦的CP值进化后数值预测，CP值的变化幅度会根据宝可梦的属性变化，于是可以考虑在模型函数中对宝可梦类别进行区分，比如：
$$
y= (b_1+ w_1 \cdot x_{cp}) \cdot \delta(X_s=Pidgey) + (b_2+ w_2 \cdot x_{cp}) \cdot \delta(X_s=Weedle) + \dots
$$
即相当于：
$$
y=b+\sum w_ix_i
$$
这样就把各个类别的模型区分开了，经过了模型的重新设计，将模型的Training Error降低到了1.9。

### 1.3 正则化

通过引入L2正则化，即权重衰减，对损失函数进行惩罚，也可以削弱过拟合。
$$
L=\sum_n(\hat y_n-(b+\sum w_ix_i))^2 + \lambda\sum(w_i)^2
$$
通过引入平方惩罚项$\lambda\sum(w_i)^2$，可以让损失函数加上权重的平方，随着权重的增加按其平方增加损失，模型训练过程中便会倾向于将$w_i$减小，宏观上会表现为分散权重，避免过于依赖某一个特征，从而避免过拟合。

对于原函数，形式变化为：
$$
y=b+\sum w_ix_i+\Delta x_i
$$
也会变得更加平滑，即倾向于对数据集中的噪声更加鲁棒。

在正则化后，训练结果优化如下：

| $\lambda$ | 0     | 1    | 10   | 100  | 1000 | 10000 | 100000 |
| --------- | ----- | ---- | ---- | ---- | ---- | ----- | ------ |
| Training  | 1.9   | 2.3  | 3.5  | 4.1  | 5.6  | 6.3   | 8.5    |
| Testing   | 102.3 | 68.7 | 25.7 | 11.1 | 12.8 | 18.7  | 26.8   |

可以看到，在引入正则化后，过拟合现象得到了显著改善，但在惩罚项的$\lambda$过大时同样会降低模型的收敛能力，因此$\lambda$也是一个需要指定的超参数。

> 参考视频节点：[ML讲座1：回归 - 案例研究](https://youtu.be/fegAeph9UaA?list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&t=4334)

## 2.分类任务

对于一个多分类任务而言，定义损失函数为：
$$
L(f)=\sum_n{\delta(f(x^n)\neq{\hat{y^n}})}
$$
即将损失函数定义为训练集中预测结果与真实类别不同的数量。

利用贝叶斯公式，对于两个种类$C_1$、$C_2$，$x$是有可能出现在$C_1$、$C_2$中的特征。$P(\left.C_1\right|x)$代表了特征$x$出现在$C_1$中的概率，$P(\left.x\right|C_1)$是代表了$C_1$中出现特征$x$的概率，$P(C_1)$代表$C_1$出现的概率，有：
$$
P(\left.C_1\right|x)=\frac{P(\left.x\right|C_1)P(C_1)}{P(\left.x\right|C_1)P(C_1)+P(\left.x\right|C_2)P(C_2)}
$$

其中，$P(\left.C1\right|x)$、$P(\left.C2\right|x)$、$P(C1)$、$P(C2)$为先验概率（Prior Probability），即根据以往经验和分析得到的概率，需要在训练集中估算出来。利用这个模型进行预测方法是：
$$
P(x)=P(\left.x\right|C_1)P(C_1)+P(\left.x\right|C_2)P(C_2)
$$

### 2.1 高斯分布

在计算先验概率时，需要对训练集中特征的分布进行拟合，这时候就可以使用高斯分布（即正态分布）作为一种拟合方式。高斯分布函数的输入为特征向量$x$，输出为$x$从这个分布中抽出的概率。
$$
f_{\mu,\Sigma}(x)=\frac{1}{(2\pi)^{\frac{D}{2}}\Sigma^{\frac{1}{2}}}\cdot e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}
$$

其中，$D$表示维度，$\mu$是$D$维期望向量，$\Sigma$是$D\times D$维协方差矩阵。从几何上来看，可以理解为$\mu$为一个点，$\Sigma$为一个范围，如果$x$离$\Sigma$越远，则概率值越小。

### 2.2 极大似然估计

不同的高斯分布对原分布的似然效果不同，而需要找到对原分布似然效果最好的一对参数$(\mu,\Sigma)$，似然值即高斯分布函数采样出原分布点的可能性，定义为$L(\mu,\Sigma)$。
$$
L(\mu,\Sigma)=f_{\mu,\Sigma}(x^1)f_{\mu,\Sigma}(x^2)\dots f_{\mu,\Sigma}(x^n)
$$

令$\mu^*,\Sigma^*=arg\underset{\mu,\Sigma}{\max}L(\mu,\Sigma)$。

在课程中宝可梦分类任务中，$\mu^*=\frac{1}{n}\sum_{i=1}^n x^i$，即均值，也可以将$L(\mu,\Sigma)$对$\mu$求偏导取零点，即取$L^{'}_{\mu}(\mu,\Sigma)=0$；$\Sigma^*=\frac{1}{n}\sum^n_{i=1}(x^i-\mu^*)(x^i-\mu^*)^T$，用这种方法就能计算出$\mu^*,\Sigma^*$。

得出$\mu^*,\Sigma^*$后，就可以对样本进行预测，计算出的$P(C_1)$就是样本$x$属于类目$C_1$的概率，取最大的概率的类为其预测结果。

在这种情况下，每个类别会存在不同的$\mu^*,\Sigma^*$，为了提高模型分类效率，尝试将$\Sigma^*$变为公共的，遵循以下的方法：
$$
\Sigma=\frac{\text{Class 1 Num}}{\text{Total}}\Sigma^1+\frac{\text{Class 2 Num}}{\text{Total}}\Sigma^2+\dots
$$
共用$\Sigma$可以增加泛化能力，减少过拟合，从而增强模型预测能力。

### 2.3 概率分布

也可以不使用高斯分布进行先验概率的估计，还可以考虑使用概率分布来进行估计，例如在二元分类问题上，可以考虑使用伯努利分布来进行估计，而在各特征独立的情况下，还可以考虑使用朴素贝叶斯分类器。

### 2.4 后验概率

后验概率（Posterior Probability），即大量试验中随机事件出现的频率逐渐稳定于其附近的某常数。

对$P(\left.C_1\right|x)$的概率分布进行展开、推导：
$$
\begin{aligned}
P(\left.C_1\right|x) &=\frac{P(\left.x\right|C_1)P(C_1)}{P(\left.x\right|C_1)P(C_1)+P(\left.x\right|C_2)P(C_2)} \\
&=\frac{1}{1+\frac{P(\left.x\right|C_2)P(C_2)}{P(\left.x\right|C_1)P(C_1)}} = \frac{1}{1+e^{-z}} = \sigma(z) \\
z &= \ln\frac{P(\left.x\right|C_1)P(C_1)}{P(\left.x\right|C_2)P(C_2)}
\end{aligned}
$$
可以把$P(\left.C_1\right|x)$拆解为关于$z$的$Sigmoid$函数表达式。
$$
\begin{aligned}
z &= \ln\frac{P(\left.x\right|C_1)}{P(\left.x\right|C_2)} \cdot\ln\frac{P(C_1)}{P(C_2)}\\
\because P(C_1) &= \frac{C_1}{C_1 + C_2} \\
P(C_2) &= \frac{C_2}{C_1 + C_2} \\
\therefore \frac{P(C_1)}{P(C_2)}&= \frac {C_1}{C_2} \\
同理可得，\ln\frac{P(\left.x\right|C_1)}{P(\left.x\right|C_2)}&=\ln\frac{|\Sigma^2|^{\frac{1}{2}}}{|\Sigma^1|^{\frac{1}{2}}}-\frac{1}{2}((x-\mu^1)^T\cdot(\Sigma^1)\cdot(x-\mu^1)-(x-\mu^2)^T\cdot(\Sigma^2)\cdot(x-\mu^2)) \\
当\Sigma^1=\Sigma^2=\Sigma时，z&=\underset{w}{\underline{(\mu^1-\mu^2)^T\cdot(\Sigma)^{-1}}}\cdot x + \underset{b}{\underline{\frac{1}{2}((\mu^1)^T\cdot\Sigma^1\cdot\mu^1-(\mu^2)^T\cdot\Sigma^2\cdot\mu^2) + \ln\frac{C_1}{C_2}}} \\
\end{aligned}
$$

模型最终被简化为了$z=wx+b$的形式，可见在共用$\Sigma$后类别的分界会变为一个线性的函数。

## 总结

在本周的学习中，对过拟合现象的出现以及解决方案进行了了解和理解，主要对L2正则化的实现方式及其效果进行了认识。同时，对于另一大类问题——分类问题的数学本质进行了推导和理解，回忆和联系了贝叶斯公式等知识， 对分类问题进行了初步了解，下周将进入逻辑回归章节的学习。
