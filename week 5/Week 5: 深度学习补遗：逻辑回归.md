[TOC]

# Week 5: 深度学习补遗：逻辑回归

## 摘要

本周继续跟随李宏毅老师的课程学习，主要对逻辑回归相关内容进行了学习和推导，对多分类任务进行了更加深入的探索。同时，针对判别型模型与生成型模型的区别进行了数学上的推导，建立了一定的认识。

## Abstract

This week, I continued to follow Professor Li Hongyi's course learning, mainly studying and deriving content related to logistic regression, and exploring multi classification tasks in more depth. At the same time, mathematical deductions were made regarding the differences between discriminative models and generative models, establishing a certain understanding.

## 1. 逻辑回归的函数变化

逻辑回归在线性回归的基础上加入了$Sigmoid$函数，使输出介于$(0,1)$之间，适用于分类问题，将线性输出结果转换为属于某个类别的概率。

![Function Set](https://i-blog.csdnimg.cn/direct/e9066c373c4d43f5bdcbedba78cbaf5a.png)

> 图片来源：(ML Lecture 5: Logistic Regression)[https://youtu.be/hSXFuypLukA?list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&t=174]

## 2. 逻辑回归的损失变化

在上一章Week 4[[Github](https://github.com/dat-G/DeepLearningStudy/blob/main/week%204/Week%204%3A%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E8%A1%A5%E9%81%97%EF%BC%9A%E5%88%86%E7%B1%BB%E4%BB%BB%E5%8A%A1.md) / [CSDN](https://blog.csdn.net/MCHacker/article/details/148817710)]中，提到利用似然函数$L(w,b)$衡量模型拟合程度的好坏。
$$
\begin{aligned}
L(w,b) &= f_{w,b}(x^1)f_{w,b}(x^2)(1-f_{w,b}(x^3))\dots f_{w,b}(x^N) \\
-\ln L(w,b) &= \ln f_{w,b}(x^1)+\ln f_{w,b}(x^2)+\ln(1-f_{w,b}(x^3))\dots +\ln f_{w,b}(x^N)\\
&=\sum_n\underset{\text{Cross Entropy between two Bernoulli distribution}}{\underline{-[\hat{y}\ln{f_{w,b}(x^n)}+(1-\hat{y})\ln{(1-f_{w,b}(x^n))}]}}
\end{aligned}
$$

经过这样的变换，可以推出两个伯努利分布之间的交叉熵。而更加一般的两个分布之间的交叉熵可以表示为：
$$
H(p,q)=-\sum_xp(x)\ln(q(x))
$$
交叉熵代表着两个分布的接近程度，两个分布越接近，其交叉熵应越接近0。因此损失函数可以表示为：
$$
\begin{aligned}
L(f)&=\sum_n C(f(x^n),\hat{y}^n) \\
C(f(x^n),\hat{y}^n) &= -[\hat{y}^n\ln f(x^n)+(1-\hat{y}^n)\ln(1-\ln f(x^n))]
\end{aligned}
$$
求出模型分布与原分布的交叉熵，即模型预测分布于原分布的相似程度。

## 3. 逻辑回归损失函数的梯度下降法数学演算

对两个伯努利分布的交叉熵，即二分类问题的损失函数对需要进行梯度下降的参数求偏导。
$$
\begin{aligned}
\frac{\partial(-\ln L(w,b))}{\partial w_i} &= \sum_n{-[\hat{y}\frac{\partial\ln{f_{w,b}(x^n)}}{\partial w_i}+(1-\hat{y})\frac{\partial\ln{(1-f_{w,b}(x^n))}}{\partial w_i}]} \\
\frac{\partial\ln{f_{w,b}(x^n)}}{\partial w_i} &= \frac{\partial\ln{f_{w,b}(x)}}{\partial z}\cdot \frac{\partial z}{\partial w_i} \quad \frac{\partial z}{\partial w_i}= x_i \\
\because f_{w,b}(x)&=\sigma(z) \quad \therefore \frac{\partial\ln{f_{w,b}(x)}}{\partial z}=\frac{\partial\ln\sigma(z)}{\partial z} = \frac{1}{\sigma(z)}\cdot\sigma'(z)=1-\sigma(z) \\
\frac{\partial\ln{(1-f_{w,b}(x^n))}}{\partial w_i} &= \frac{\partial\ln{(1-f_{w,b}(x^n))}}{\partial z} \frac{\partial z}{\partial w_i}=\frac{\partial\ln{(1-\sigma(z))}}{\partial z} \cdot x_i = \frac{1}{1-\sigma(z)}\cdot\sigma'(z)=\sigma(z)\cdot x_i \\
因此，\frac{\partial(-\ln L(w,b))}{\partial w_i} &= \sum_n -[\hat{y}^n(1-f_{w,b}(x^n))x_i^n-(1-\hat{y}^n)f_{w,b}(x^n)x_i^n] \\
&=\sum_n -[\hat{y}^n-\hat{y}^nf_{w,b}(x^n)-f_{w,b}(x^n)+\hat{y}^nf_{w,b}(x^n)]x_i^n \\
&=\sum_n -[\hat{y}^n-f_{w,b}(x^n)]x_i^n
\end{aligned}
$$

这样就求出了$-\ln L(w,b)$对$w_i$的梯度，通过$w_i\gets w_i-\eta\sum_n -[\hat{y}^n-f_{w,b}(x^n)]x_i^n$就可以对$w_i$进行更新。

## 4. 判别型模型 Discriminative Model 与 生成型模型 Generative Model

对于概率模型$P(\left.C_1\right| x)=\sigma(w\cdot x+b)$，判别型模型通过反向传播直接拟合权重$w$与偏差$b$，而生成型模型通过假设概率分布符合高斯分布等分布模型，首先找出$\mu^1$，$\mu^2$，$\Sigma^{-1}$，则有：
$$
\begin{aligned}
w^T &= (\mu^1-\mu^2)^T\Sigma^{-1} \\
b &= -\frac{1}{2}(\mu^1)^T(\Sigma^1)^{-1}\mu^1+\frac{1}{2}(\mu^2)^T(\Sigma^2)^{-1}\mu^2+\ln\frac{N_1}{N_2}
\end{aligned}
$$
在李宏毅老师的[课程](https://youtu.be/hSXFuypLukA?list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&t=2161)中，提到了对于生成式模型，取得了73%的正确率，而判别型模型79%，同时提到了**判别型模型通常会比生成性模型表现更好**。

生成式模型的优势主要有：

- 通过对概率分布的估计，减少了对训练集数据的要求

- 通过对概率分布的估计，对噪声更加鲁棒

- 将先验概率与类条件概率分开，可以从不同来源估计

  （例如，在翻译模型中，先验概率为文字出现概率，可以从纯文本中获取，而类条件概率中才需要出现语音与文字联系，减少训练难度）

## 5. 多类别分类模型的输出

假设有三个类别，输出分别为$z_1=w^1\cdot x+b^1$、$z_2=w^2\cdot x+b^2$、$z_3=w^3\cdot x+b^3$。

使用$Softmax$对其进行归一化输出类别概率，如：
$$
\begin{bmatrix}
z_1 \\
z_2 \\
z_3
\end{bmatrix}
\to
\begin{bmatrix}
e^{z_1} \\
e^{z_2} \\
e^{z_3} 
\end{bmatrix}
\to
\sum_{j=1}^3 z_j
\to
\begin{bmatrix}
y_1=\frac{e^{z_1}}{\sum_{j=1}^3 z_j} \\
y_2=\frac{e^{z_2}}{\sum_{j=1}^3 z_j} \\
y_3=\frac{e^{z_3}}{\sum_{j=1}^3 z_j} \\
\end{bmatrix}
$$
这样就求得了归一化的类别概率。

## 6. 多类别分类模型的交叉熵损失

$$
\begin{bmatrix}
y_1 \\
y_2 \\
y_3
\end{bmatrix}
\overset{\text{Cross Entropy}}{\underset{\sum_{i=1}^3\hat{y_i}\ln y_i}{\longleftrightarrow}}
\begin{bmatrix}
\hat{y_1} \\
\hat{y_2} \\
\hat{y_3}
\end{bmatrix}
$$

而其真实值向量，例如$x\in \text{Class 1}$时，$\hat{y}=\begin{bmatrix}1\\0\\0\end{bmatrix}$，以此类推，就可以计算出预测结果的交叉熵损失。

## 7. 逻辑回归的限制

![Limitation of Logistic Regression](https://i-blog.csdnimg.cn/direct/f05c22b4bf84483b9e7a93e6ac975887.png)

> 图片来源：(ML Lecture 5: Logistic Regression)[https://youtu.be/hSXFuypLukA?list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&t=3449]

可见，因为逻辑回归的类别边界为平面上的一条分界线，而如图的情况中无法有一条线能划分开两个类别，就需要进行特征变换。

![Feature Transformation](https://i-blog.csdnimg.cn/direct/f54c49a7c35146a5bbfeb1b4727af31b.png)

在这个例子中，即将输入的$x$、$y$坐标映射为坐标点到$\begin{bmatrix}0\\0\end{bmatrix}$与$\begin{bmatrix}1\\1\end{bmatrix}$距离的向量，就完成了特征变换，但寻找一个好的变换并不总是那么容易。

## 总结

在本周的学习中，我学习到了分类任务中的逻辑回归与极大似然中的联系，同时对于判别型与生成型两种模型进行了推导与探讨，理解了两者之间的区别与优劣。接着，对交叉熵这种分类任务中常用的损失函数进行了推导与计算，对其计算逻辑产生了基本的认识。最后，通过对逻辑回归限制的探讨，引出了对特征变换的认识。
