# Week 17: 深度学习补遗：Boosting和量子逻辑门

## 摘要

本周继续跟随李宏毅老师的课程，学习了上周的Adaboost训练分类器后的集成方法，同时学习了量子计算相关进阶内容。对Boosting的一般方法进行了学习，同时对量子逻辑门的基本概念有了一定的了解。

## Abstract

This week, I continued following Professor Hung-yi Lee's course, learning ensemble methods following last week's study of Adaboost training classifiers, while also covering advanced topics in quantum computing. I studied the general methodology of Boosting and gained a foundational understanding of the basic concepts of quantum logic gates.

## 1. Adaboost聚合分类器的方法

对于训练出的$T$个分类器$f_1(x),\dots,f_t(x),\dots,f_T(x)$，需要经过下述步骤聚合成一个模型。

- 统一权重
  - $H(x)=sign(\sum_{t=1}^Tf_t(x))$
- 非统一权重
  - $H(x)=sign(\sum_{t=1}^T\alpha_tf_t(x))$

即，当$H(x)\geq0$时，属于类别1；而当$H(x)<0$时，属于类别2。而其中，$\alpha_t=\ln\sqrt{(1-\epsilon_t)/\epsilon_t}$，$u^n_{t+1}= u_t^n\times e^{-\hat{y^n}f_t(x^n)\alpha_t}$。简单来说，更小的误差会给分类器带来更大的权重。在合并分类器时，统一权重的方法不好，因为分类器有强和弱的分别，用非统一权重的方式更符合直觉。

## 2. Gradient Boosting方法

Adaboost是Boosting方法中的一个特例，普通的Boosting方法如下。

- 初始化函数$g_0(x)=0$
- 从$t=1$到$T$，找到一个函数$f_t(x)$和$\alpha_t$来提升$g_{t-1}(x)$，$g_{t-1}(x)=\sum_{i=1}^{t-1}a_if_i(x)$
- $g_t(x)=g_{t-1}(x)+\alpha_tf_t(x)$
- 输出$Output=H(x)=sign(g_T(x))$

学习目标是最小化损失$L(g)=\sum_nl(\hat{y^n},g(x^n))=\sum_n e^{-\hat{y^n}g(x^n)}$。
$$
g_t(x)=g_{t-1}(x)-\left.\eta\frac{\partial L(g)}{\partial g(x)}\right|_{g_t(x)=g_{t-1}(x)}
$$
利用梯度下降最小化损失函数，并且希望$-\left.\eta\frac{\partial L(g)}{\partial g(x)}\right|_{g_t(x)=g_{t-1}(x)}$与$\alpha_tf_t(x)$同向。

如果把$\sum_n e^{-\hat{y^n}g(x^n)}$作为损失函数的话，实际上求解的$f_t$、$\alpha_t$就是Adaboost求解的结果，但是Gradient Boost可以修改损失函数，作为更一般的解决方案来解决问题。

## 3. 量子计算初步

在描述多个粒子的量子态时，可以写作$\ket{01}$，也可以写作$\ket{0}\otimes\ket{1}$或$\ket{0}\ket{1}$，并且可以运用分类率把多个量子比特的状态像乘法一样进行运算。
$$
\begin{aligned}
\ket{S}&=\sqrt{\frac{1}{10}}\ket{00}+\sqrt{\frac{1}{10}}\ket{01}-\sqrt{\frac{2}{5}}\ket{10}-\sqrt{\frac{2}{5}}\ket{11} \\
\ket{S}&=\sqrt{\frac{1}{5}}\ket{0}\otimes(\sqrt{\frac{1}{2}}\ket{0}+\sqrt{\frac{1}{2}}\ket{1})-\sqrt{\frac{4}{5}}\ket{1}\otimes(\sqrt{\frac{1}{2}}\ket{0}+\sqrt{\frac{1}{2}}\ket{1})\\
\ket{S}&=(\sqrt{\frac{1}{5}}\ket{0}-\sqrt{\frac{4}{5}}\ket{1})\otimes(\sqrt{\frac{1}{2}}\ket{0}+\sqrt{\frac{1}{2}}\ket{1})
\end{aligned}
$$
因此，两个粒子的量子态就可以模拟四个维度上的空间的概率分布，即$n$个电子就可以模拟$2^n$个维度上的概率分布，通常来说，需要记录这么多概率的数字，需要$2^n$个单元。但实际上$n$个量子比特并不能完全替代$2^n$个经典比特，因为量子比特实际上并没有“存储”概率分布的数字，无法输出单一维度的概率，我们能做的只有测量，并且以本征态系数平方的形式输出。

## 4. 量子逻辑门

量子逻辑门对比经典逻辑门的一个显著区别是，经典逻辑门可以输出数量和输入数量不同，但量子逻辑门的输出数量和输入数量永远一致。

### 4.1 Hadamard门

哈达玛门对输入做了一个态向量的反射， 关于与$x$轴夹角为22.5度的一条对称轴进行输入的反射，任何输入的量子态都会被作用到反射之后的位置上。 
$$
\ket{0}\to Hadamard\to \frac{\ket{0}+\ket{1}}{\sqrt{2}} \\
\ket{1}\to Hadamard\to \frac{\ket{0}-\ket{1}}{\sqrt{2}} \\
$$

### 4.2 CNOT门

CNOT门由两个输入构成，分别是控制位和目标位：假如控制位为0，那么直接将目标位输出；假如控制位为1，那么就翻转目标位。
$$
\ket{0}\otimes\ket{1}\to CNOT \to \ket{0}\otimes\ket{1} \\
\ket{1}\otimes\ket{0}\to CNOT \to \ket{1}\otimes\ket{1} \\
$$
量子逻辑门和经典逻辑门的一个最大的本质区别是，**量子逻辑门的每个输入实际上是一个叠加态，对每一个输入的处理实际上是将其每一个本征态分别操作，变为其新的本征态，再进行叠加**。
$$
a\ket{00}+b\ket{01}+c\ket{10}+d\ket{11}\to CNOT\to a\ket{00}+b\ket{01}+d\ket{10}+c\ket{11}
$$
这个特性叫做量子门的线性叠加，量子门必须保证输出的系数是来自某些输入的系数的线性叠加，同时，输入和输出所有系数的平方和保持为1，数学上称为幺正变换 。

## 总结

本周对Boosting相关内容进行了收尾，对Adaboost训练出的多个分类器进行聚合的方法进行了了解，同时对Boosting的一般化方法Gradient Boosting进行了学习。最后，对量子逻辑门的概念进行了了解，其区别与普通逻辑门一系列特性，预计下周继续推进量子计算相关学习，在深度学习方面继续学习Stacking相关知识。