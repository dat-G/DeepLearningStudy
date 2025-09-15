[TOC]

# Week 16: 深度学习补遗：集成学习进阶与量子计算概念入门

## 摘要

本周跟随李宏毅老师的课程继续学习了Adaboost相关的知识和数学原理推导，同时根据老师的指引开始补充部分量子计算相关的基础知识。

## Abstract

This week, I continued studying Adaboost-related knowledge and mathematical principle derivations in Professor Li Hongyi's course. At the same time, following the professor's guidance, I began supplementing some foundational knowledge related to quantum computing.

## 1. Adaboost

Adaboost是一种Boosting的方法，其具体做法是，训练好一个分类器$f_1$后，调整训练集的权重，让$f_1$的正确率在调整后的训练集上降低到50%的正确率。

在调整训练集权重时，Adaboost采用的方法是，对于$f_1$分类正确的数据，让其权重除以一个大于一的数值$d_1$；同时对于$f_1$错分的数据，让其权重乘以同样的一个数值$d_1$。
$$
\epsilon_1
=\frac{\sum_n u_1^n\delta(f_1(x^n)\neq\hat{y^n})}{\sum_n u_1^n}
=\frac{\sum_n u_2^n\delta(f_1(x^n)\neq\hat{y^n})}{\sum_n u_2^n} \\
\sum_n u_2^n\delta(f_1(x^n)\neq\hat{y^n})=\sum_{f_1(x^n)\neq\hat{y^n}}u_1^nd_1\\
\sum_nu_2^n=\sum_{f_1(x^n)\neq\hat{y^n}}u_1^nd_1+\sum_{f_1(x^n)=\hat{y^n}}u_1^n/d_1
$$
错误率$\epsilon_1$的计算方式是，将错误数据的权重求和，并且进行归一化。使$f_1$的正确率降低到50%具体来说就是，在$f_1$在$u_1$的$\epsilon_1\textless 0.5$时，调整权重得到$u_2$使得$f_1$在$u_2$上的$\epsilon_2=0.5$，并在权重$u_2$上训练$f_2$。
$$
\begin{aligned}
\because\frac{\sum_{f_1(x^n)\neq\hat{y^n}}u_1^nd_1}{\sum_{f_1(x^n)\neq\hat{y^n}}u_1^nd_1+\sum_{f_1(x^n)=\hat{y^n}}u_1^n/d_1}&=0.5 \\
\therefore \frac{\sum_{f_1(x^n)\neq\hat{y^n}}u_1^nd_1+\sum_{f_1(x^n)=\hat{y^n}}u_1^n/d_1}{\sum_{f_1(x^n)\neq\hat{y^n}}u_1^nd_1}&=2\\
\frac{\sum_{f_1(x^n)=\hat{y^n}}u_1^n/d_1}{\sum_{f_1(x^n)\neq\hat{y^n}}u_1^nd_1}=1
\end{aligned}
$$
这就说明$\sum_{f_1(x^n)=\hat{y^n}}u_1^n/d_1=\sum_{f_1(x^n)\neq\hat{y^n}}u_1^nd_1$，即答对部分新的权重的总和需要等于答错部分新的权重总和。
$$
\begin{aligned}
\sum_{f_1(x^n)=\hat{y^n}}u_1^n/d_1&=\sum_{f_1(x^n)\neq\hat{y^n}}u_1^nd_1 \\
\frac{1}{d_1}\sum_{f_1(x^n)=\hat{y^n}}u_1^n&=d_1\sum_{f_1(x^n)\neq\hat{y^n}}u_1^nd_1\\
\because \epsilon_1
&=\frac{\sum_n u_1^n\delta(f_1(x^n)\neq\hat{y^n})}{\sum_n u_1^n}\\
\therefore \frac{\sum_n u_1^n(1-\epsilon_1)}{d_1}&=\sum_n u_1^n\epsilon_1d_1\\
\therefore d_1=\sqrt{(1-\epsilon_1)/\epsilon_1}>1
\end{aligned}
$$

可以得到，用这个$d_1$去乘以或者除$u_1$，就可以生成一个可以让$f_1$Fail掉的训练集。

可以将Adaboost的步骤整理。

- 对于训练数据$\{(x^1,\hat{y^1},u_1^1),\dots,(x^n,\hat{y^n},u_1^n),\dots,(x^N,\hat{y^N},u_1^N)\}$，其中$\hat{y}=\pm1$（二元分类），$u_1^n=1$（初始化为等权重）
- 对于$t=1,\dots,T$：
  - 用训练集权重$\{u^1_t,\dots,u_t^N\}$训练一个较弱的分类器$f_t(x)$，得到其分类错误率$\epsilon_t$。
  - 对于$n=1,\dots,N$：
    - 如果$x^n$被$f_t(x)$错分了，即$\hat{y^n}\neq f_t(x^n)$，那么训练集权重$u^n_{t+1}=u^n_t\times d_t=u_t^n\times e^{\alpha_t}$。
    - 对于分类正确的情况，即$\hat{y^n}= f_t(x^n)$，那么$u^n_{t+1}=\frac{u^n_t}{d_t}=u^n_t\times e^{-\alpha^t}$。
    - 其中，$d_t=\sqrt{(1-\epsilon_1)/\epsilon_1}$，$\alpha_t=\ln\sqrt{(1-\epsilon_1)/\epsilon_1}$
    - 即$u^n_{t+1}\gets u_t^n\times e^{-\hat{y^n}f_t(x^n)\alpha_t}$

## 2. 量子计算初步

一个集合如果满足以下运算法则就能被称为一个矢量。

- 加法结合律：$\bold{u}+(\bold{v}+\bold{w})=(\bold{u}+\bold{v})+\bold{w}$
- 加法交换律：$\bold{u}+\bold{v}=\bold{v}+\bold{u}$
- 加法幺元：存在 $\bold{0} \in \it{V}$，称为0矢量，存在对于任意$\bold{v}$有$\bold{v}+\bold{0}=\bold{v}$
- 加法逆元：对于任意$\bold{v}$，存在 $\bold{-v} \in \it{V}$，称为$\bold{v}$的逆元，令$\bold{v}+(\bold{-v})=0$
- 标量乘法与域乘法兼容性：$a(b\bold{v})=(ab)\bold{v}$
- 标量乘法幺元：$\bold{1v}=\bold{v}$，$\bold{1}$是$F$中的乘法单位元。
- 标量乘法对向量加法的分配率：$a(\bold{u+v})=a\bold{u}+a\bold{v}$
- 标量乘法对域加法的分配率：$(a+b)\bold{v}=a\bold{v}+b\bold{v}$

例如一个由“波函数”组成的集合，比如：$H=\{\text{Spin up, Spin down }\}$，描述了电子自旋的状态。使用狄拉克表示，我们可以将其表示成：$\ket{0}=\text{Spin up},\ket{1}=\text{Spin down}$。$\ket{0}$和$\ket{1}$就被称为“量子比特”（Qubit）。这种表示方式是由狄拉克发明的，分别称为$\bra{bra}$和$\ket{ket}$。一个电子的状态就可以表示为$a\ket{0}+b\ket{1}$，其中$|a|^2+|b|^2=1(a,b\in \mathbb{C})$。

## 总结

本周机器学习方面，学习了Adaboost相关知识和推导，理解了Adaboost更新训练集权重组成新训练集以及利用训练多个分类器提升训练集准确率的方法。学习了量子计算的基本概念，关于矢量定义的相关内容。