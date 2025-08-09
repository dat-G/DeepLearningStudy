[TOC]

# Week 11: 深度学习补遗：支持向量机

## 摘要

本周主要继续跟进李宏毅老师的进度，学习支持向量机相关的知识，研究其底层数学原理与数学推导。

## Abstract

This week, we will continue to follow up on Mr. Hung-yi Lee's progress, learning about support vector machines and studying their underlying mathematical principles and mathematical derivations.

## 1.Support Vector Machine 支持向量机

> "Hinge Loss + Kernel Tree = Support Vector Machine"

对于一个基本的二元分类模型，可以描述如下：
$$
Data: \\
\begin{bmatrix}
x^1&&x^2&&x^3&&\dots \\
\hat{y}^1&&\hat{y}^2&&\hat{y}^3&&\dots
\end{bmatrix} \\
\hat{y}^n=+1,-1
$$


- 1. 函数建模

$$
g(x)=\left\{
\begin{aligned}
f(x)>0\quad Output=+1 \\
f(x)<0\quad Output=-1
\end{aligned}
\right.
$$

- 2. 损失函数

$$
L(f)=\sum_n \delta (g(x^n)\neq \hat{y}^n)
$$

因为损失函数不可微，因此这个模型无法用梯度下降法求解。所以，我们要考虑用一个可微的函数来替代这个理想的损失函数。
$$
\begin{aligned}
\text{Ideal Loss: }L(f)&=\sum_n\delta (g(x^n)\neq \hat{y}^n) \\
\text{Approximation: }L(f)&=\sum_nl(f(x^n),\hat{y}^n)
\end{aligned}
$$
比如，平方损失，即如果$\hat{y}^n=1$，那么$f(x)$接近1；如果$\hat{y}^n=-1$，$f(x)$接近负一，可以描述为$l(f(x^n),\hat{y}^n)=(\hat{y}^nf(x^n)-1)^2$。简单来说，就是当$\hat{y}^n=1$时，$l(f(x^n),\hat{y}^n)=(f(x^n)-1)^2$；当$\hat{y}^n=-1$时，$l(f(x^n),\hat{y}^n)=(-f(x^n)-1)^2$。
$$
\begin{aligned}
\text{Square Loss: }l(f(x^n),\hat{y}^n)&=(\hat{y}^nf(x^n)-1)^2 \\
&=\left\{
\begin{aligned}
(f(x^n)-1)^2\quad &\text{while }\hat{y}^n=1 \\
(-f(x^n)-1)^2\quad &\text{while }\hat{y}^n=-1
\end{aligned}
\right.
\end{aligned}
$$
也可以选择Sigmoid+平方损失。
$$
\begin{aligned}
\text{Square Loss: }l(f(x^n),\hat{y}^n)&=(\sigma(\hat{y}^nf(x^n))-1)^2 \\
&=\left\{
\begin{aligned}
(\sigma(f(x^n))-1)^2\quad &\text{while }\hat{y}^n=1 \\
(1-\sigma(f(x^n))-1)^2\quad&\\
=(\sigma(f(x)))^2\quad &\text{while }\hat{y}^n=-1
\end{aligned}
\right.
\end{aligned}
$$
因为在逻辑回归的实际实践中采用的是交叉熵，因此还可以选用Sigmoid+交叉熵的损失，衡量真实分布以及预测分布之间的交叉熵。
$$
\text{Square Loss: }l(f(x^n),\hat{y}^n)=\ln (1+e^{-\hat{y}^n f(x)})
$$
这时候就可以引入Hinge Loss，Hinge Loss的表达式为$l(f(x^n,\hat{y^n}))=\max(0,1-\hat{y^n}f(x))$，其设计思想为，对于一个$\hat{y^n}=1$的例子，只需要让$f(x)>1$就是已经满意了，不需要进行优化了；同理，对于一个$\hat{y^n}=-1$的例子来说，只需要让$f(x)<-1$就完美了。Hinge Loss实际上和交叉熵比较相似，而Hinge Loss对比交叉熵损失来说，最显著的优势是其对于异常值更加鲁棒，在异常值比较多的情况下，Hinge Loss要显著优于交叉熵。

对于线性的支持向量机，可以描述如下：

- 1. 函数建模


$$
f(x)=\sum_i w_ix_i+b=
\begin{bmatrix}
w \\ b
\end{bmatrix}
\cdot
\begin{bmatrix}
x\\1
\end{bmatrix}
=w^Tx
$$

- 2. 损失函数

$$
L(f)=\sum_n l(f(x^n),\hat{y^n})+\lambda||w||_2 \\
l(f(x^n,\hat{y^n}))=\max(0,1-\hat{y^n}f(x))
$$

实际上，SVM和逻辑回归最主要的差别就是损失函数，SVM使用了Hinge Loss，而逻辑回归采用了交叉熵损失函数。虽然Hinge Loss在某点不可微，但是实际上类似ReLU等的流行损失函数同样会在某点不可微，然而实际上他们仍然可以运用梯度下降法进行训练。 

## 2. 线性支持向量机的梯度下降推导

$$
L(f)=\sum_n l(f(x^n),\hat{y^n}) \quad l(f(x^n,\hat{y^n}))=\max(0,1-\hat{y^n}f(x)) \\
\frac{\partial l(f(x^n,\hat{y^n}))}{\partial w_i} = \frac{\partial l(f(x^n,\hat{y^n}))}{\partial f(x^n)} \frac{\partial f(x^n)}{\partial w_i} \\
\because f(x^n)=w^T\cdot x^n \quad\frac{\partial \max(0,1-\hat{y^n}f(x^n))}{\partial  f(x^n)}=\left\{
\begin{aligned}
-\hat{y^n}\quad& \text{If }\hat{y^n}f(x^n)<1 \\
&1-\hat{y^n}f(x^n)>1\\
0\quad&\text{Otherwise}
\end{aligned}
\right. \\
\therefore \frac{\partial L(f)}{\partial w_i}=\sum_n\underset{c^n(w)}{\underline{-\delta(\hat{y^n}f(x^n)<1)\hat{y^n}x^n_i}} \\
\text{Update: }w_i\gets w_i -\eta\sum_n c^n(w)x_i^n
$$

这样就完成了线性SVM的一次梯度下降，而线性SVM还可以用另一种表示方法，避免使用$\max$函数。
$$
L(f)=\sum_n \underset{\epsilon^n}{\underline{l(f(x^n),\hat{y^n})}}+\lambda||w||_2 \\
\because \epsilon^n=\max(0,1-\hat{y^n}f(x)) \\
\therefore \epsilon^n \geq0 \quad \epsilon^n\geq1-\hat{y^n}f(x) \rightarrow \hat{y^n}f(x)\geq1-\epsilon^n
$$
虽然$\epsilon^n=\max(0,1-\hat{y^n}f(x))$和$\left\{\begin{aligned}\epsilon^n &\geq0 \\ \hat{y^n}f(x))&\geq1-\epsilon^n\end{aligned}\right.$不等价，即后者带入任何一个无穷大都会满足条件，但是在梯度下降需要最小化$L(f)$的背景下，其是等价的。在新的表达式下，式子可以简单被解释为，要求为同号的$\hat{y^n}$和$f(x)$相乘后为1，可以放宽一个大于零的松弛变量$\epsilon$。

## 3. Kernel Method 核方法

在梯度下降中的更新式子中，$\text{Update: }w_i\gets w_i -\eta\sum_n c^n(w)x_i^n$中的$c^n(w)$指的是$\frac{\partial l(f(x^n,\hat{y^n}))}{\partial w_i}$。但Hinge Loss类似ReLU，是个分段函数，在$max=0$的情况下，$c^n(w)$常常是0。所以，线性组合的权重是稀疏的，不是所有数据都会被加到结果上，权重不为0的数据点就被称为支持向量。

因为采用了Hinge Loss，其权重向量具有稀疏性，因此对比其他方法来说，支持向量机更加鲁棒。

将$w$写成数据点的线性组合$w=\sum_n \alpha_n x^n$，$X=[x^1\quad x^2 \quad \dots\quad x^N]$，$\alpha=\begin{bmatrix}\alpha_1\\\alpha_2\\\vdots\\\alpha_N\end{bmatrix}$。因此，$f(x)=w^Tx\rightarrow f(x)=\alpha^TX^Tx$，即$f(x)=\sum_n\alpha_n(x^n\cdot x)=\sum_n a_nK(x^n,x)$，问题就转变成需要寻找一组$\{\alpha_1^*,\dots,a_n^*,\dots,\alpha_N^*\}$最小化损失函数，损失函数也可以变成与Kernel函数对应的表达形式。
$$
L(f)=\sum_n l(f(x^n),\hat{y^n})=\sum_nl(\sum_{n'}\alpha_{n'}K(x^{n'},x^n),\hat{y^n})
$$
可以观察到，实际上我们不再需要知道整个向量$x$，而是只需要知道$x$和$z$的内积值，这个就叫做Kernel Trick。假如$x=\begin{bmatrix}x_1\\ x_2\end{bmatrix}$，可以进行特征导出$\phi(x)=\begin{bmatrix}x_1^2\\\sqrt{2}x_1x_2\\ x_2^2\end{bmatrix}$。
$$
\begin{aligned}
K(x,z)&=\phi(x)\cdot\phi(z)=
\begin{bmatrix}
x_1^2\\
\sqrt{2}x_1x_2\\
x_2^2
\end{bmatrix}
\cdot
\begin{bmatrix}
z_1^2\\
\sqrt{2}z_1z_2\\
z_2^2
\end{bmatrix} \\
&=x_1^2z_1^2+2x_1x_2z_1z_2+x_2^2z_2^2 \\
&=(\begin{bmatrix}
x_1\\
x_2
\end{bmatrix}
\cdot
\begin{bmatrix}
z_1\\
z_2
\end{bmatrix})=(x\cdot z)^2
\end{aligned}
$$

而直接运算$K(x,z)=(x\cdot z)^2$实际上只需要计算$k$维，但假如先进行高维映射，转换到$\phi(x)\cdot\phi(z)$，就需要进行$k^2$维运算，因此，Kernel Trick在简化运算上实际有比较重要的作用。

##总结

本周对支持向量机相关知识进行了学习，着重学习了支持向量机与逻辑回归的关键区别Hinge Loss及其数学推导部分。同时，在抽象层面上了解了支持向量机权重的稀疏性以及其设计理念，对这一经典模型设计有了一定的认识。