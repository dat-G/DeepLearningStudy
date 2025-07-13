[TOC]

# Week 7: 深度学习补遗：线性方法的无监督学习

## 摘要

本周继续跟随李宏毅老师的课程进行学习，主要深入学习了两种线性无监督学习聚类方法，K-means和PCA。同时，对PCA的降维方法的数学原理进行了详细的总结和推导。

## Abstract

This week, I continued to follow Professor Hung-yi Lee's course and mainly delved into two linear unsupervised learning clustering methods, K-means and PCA. At the same time, a detailed summary and derivation of the mathematical principles of PCA dimensionality reduction methods were conducted.

## 1. 聚类 Clustering

聚类的一个目的，是降低维度，可以把复杂维度的输入转换为简单维度的输出。K-means是一种常用的聚类方法。

- K-means聚类步骤
  1. 将样本$X=\{x^1, \dots, x^n, \dots, x^N\}$聚类成$K$个类别。
  
  2. 初始化聚类中心$c^i$，$i=1,2,\dots,K$，$K$个聚类中心$c^i$从$X$中随机取出。
  
  3. 重复过程
     1. 对于所有的$X$中所有的$x^n$：$b^i\left\{\begin{aligned}1&\quad x^n是c^i的k近邻\\0&\quad其他\end{aligned}\right.$
     
     2. 更新所有的$c^i$：
     
        $$
        c^i=\frac{\sum_{x^n}b_i^nx^n}{\sum_{x^n}b_i^n}
        $$
  
- 层次聚类 Hierarchical Agglomerative Cluster 步骤

  1. 建树，重复步骤直到森林变为唯一的树：
     1. 计算最高节点之间的相似度
     2. 合并相似度最高的节点，建立其共同父亲，需要根据节点之间距离划定高度
  2. 选择一个阈值，对树进行切割，分成几个聚类。

## 2. 分布式表示 Distributed Representation

聚类则一个样本必须属于且只属于一个类别，分布式表示指的是可以显示对不同类别的倾向性。 

例如，在聚类分析下，会得出小杰是强化系的结论。但在分布式表示下，就会得出一个关于小杰的分布式表示形式，更符合实际情况。

| 强化系   | 0.70 |
| -------- | ---- |
| 放出系   | 0.25 |
| 变化系   | 0.05 |
| 操作系   | 0.00 |
| 具现化系 | 0.00 |
| 特质系   | 0.00 |

## 3. 降维 Dimension Reduction

大部分表示并不需要原维度那么高的维度就可以对信息进行表示，因此可以进行降维操作节省内存和算力。
$$
\underset{\vec{z}的维度小于\vec{x}}{\vec{x}\to function\to\vec{z}}
$$
在降维上可以选用几种不同的方法：

- 特征选择 Feature Selection：对不必要的特征进行直接去除（很多时候不见得有用）

- 主成分分析 Principle Component Analysis：

  - $z=Wx$，减到一维，$z_1=w^1\cdot x$，即将$x$中所有数据点通过矩阵$W$投影到$z$。

  - 需要限制矩阵的行的二范数（2-norm）为1，即$||w^1||_2=\sqrt{(w^1_1)^2+\dots}=1$，即$z=Wx$可以被解释称投影的形式。

  - 在投影时，希望其分布更大，即更好的体现其特征。
    $$
    \begin{aligned}
    arg \max Var(z_1)&=\sum_{z_1}(z_1-\bar{z_1}) \\
    arg \max Var(z_2)&=\sum_{z_2}(z_2-\bar{z_2}) \\
    w_1\cdot w_2&=0 \\
    \dots
    \end{aligned}
    $$
    因此，找出来的$W=\begin{bmatrix}(w^1)^T\\(w^2)^T\\\dots\end{bmatrix}$会是一个正交矩阵。
    $$
    \bar{z_1}=\frac{1}{n}\sum z_1=\frac{1}{n}\sum w^1\cdot x=w^1\cdot\frac{1}{n}\sum x=w^1\cdot\bar{x}
    $$
    
    $$
    \begin{aligned}
    Var(z_1)&=\sum_{z_1}(z_1-\bar{z_1}) \\
    &=\sum_x(w^1\cdot x-w^1\cdot\bar{x}) \\
    &=\sum(w^1\cdot(x-\bar{x}))^2 \\
    &=\sum(w^1)^T(x-\bar{x})(x-\bar{x})^Tw^1 \\
    &=(w^1)^T\underset{Covariance}{\underline{\sum(x-\bar{x})(x-\bar{x})^T}}w^1 \\
    &=(w^1)^TCov(x)w^1 \\
    \because||w^1||_2&=(w^1)^Tw^1=1
    \end{aligned}
    $$
		$Cov(x)$是一个对称且半正定的矩阵，即其所有特征值都是非负的。运用拉格朗日乘子法 Lagrange Multiplier：
		$$
		g(w^1)=(w^1)^TSw^1-\alpha((w^1)^Tw^1-1) \\
		\left.
		\begin{aligned}
		\frac{\partial g(w^1)}{\partial w^1_1}& = 0 \\
		\frac{\partial g(w^1)}{\partial w^1_2} &= 0 \\
		\dots
		\end{aligned}
		\right\}
		\begin{aligned}
		&Sw^1-\alpha w^1=0\\
		&Sw^1=\alpha w^1 \\
		&(w^1)^TSw^1=\alpha(w^1)^Tw^1
		\end{aligned}
		$$
		有结论：当$w^1$对应到最大的特征值的特征向量时，可以让$\alpha$最大，为最大的特征值$\lambda_1$。
		
		同样，找到使得根据$w_2$投影之后分布$(w^{2})^{T}Sw^{2}$最大的$w_2$。
		$$
		\left\{
		\begin{aligned}
		(w^{2})^{T}w^{2}=1 \\
		(w^{2})^{T}w^{1}=0 \\
		\end{aligned}
		\right. \\
		$$
		
		$$
		g(w^{2})=(w^{2})^{T}Sw^{2}-\alpha\left((w^{2})^{T}w^{2}-1\right)-\beta\left((w^{2})^{T}w^{1}-0\right) \\
		\left.
		\begin{aligned}
		\frac{\partial g(w^2)}{\partial w^2_1}& = 0 \\
		\frac{\partial g(w^2)}{\partial w^2_2} &= 0 \\
		\dots
		\end{aligned}
		\right\}
		\begin{aligned}
		Sw^2-\alpha w^2-\beta w^1&=0\\
		\underset{0}{\underline{(w^1)^TSw^2}}-\alpha\underset{0}{\underline{(w^1)^Tw^2}}-\beta \underset{1}{\underline{(w^1)^Tw^1}}&=0 \\
		=((w^1)^TSw^2)^T &=(w^2)^TS^Tw^1 \\
		=(w^2)^TSw^1 &=\lambda_1(w^2)^Tw^1 = 0\quad(Sw^1=\lambda_1w^1)
		\end{aligned} \\
		\beta=0:Sw^2-\alpha w^2=0\quad Sw^2=\alpha w^2
		$$
		将原数据分布映射乘对角阵，可以压缩特征。
		
		![PCA](https://i-blog.csdnimg.cn/direct/5c635a03e5164b3683562c879e94c474.png)
		$$
		\begin{aligned}
		 & Cov(z)=\sum(z-\bar{z})(z-\bar{z})^{T}=WSW^{T}\quad S=Cov(x) \\
		 & =WS\left[
		\begin{array}
		{ccc}{W^{1}} & {\cdots} & {W^{K}}
		\end{array}\right]=W\left[
		\begin{array}
		{ccc}{S_{W^{1}}} & {\cdots} & {S_{W^{K}}}
		\end{array}\right] \\
		 & =W[\lambda_{1}w^{1}\quad\cdots\quad\lambda_{K}w^{K}]\quad=[\lambda_{1}Ww^{1}\quad\cdots\quad\lambda_{K}Ww^{K}] \\
		 & =
		\begin{bmatrix}
		\lambda_{1}e_{1} & \cdots & \lambda_{K}e_{K}
		\end{bmatrix}=D\quad\text{[Diagonal matrix]}
		\end{aligned}
		$$
		更加形象化的来说，对于一张图片$x$及构成其的部件$u_i$来说，有：
		$$
		x\approx c_1u^1+c_2u^2+\dots+c_Ku^K+\bar{x}
		$$
		因此，$\begin{bmatrix}c_1\\c_2\\\dots\\c_K\end{bmatrix}$是描述一张图片的有效表示。易知，$x$减所有图像的平均$\bar{x}$可以表示为部件$u_i$的线性组合。
		$$
		x-\bar{x}\approx c_1u^1+c_2u^2+\dots+c_Ku^K=\hat{x} \\
		\text{Reconstruction Error}=||(x-\bar{x})-\hat{x}||_2
		$$
		可以得到用图片组件线性表示整张图片的重建误差，需要找到$\{u^1,\dots,u^K\}$最小化误差。
		$$
		L=\underset{\{u^1,\dots,u^K\}}{\min}\sum||(x-\bar{x})-\underset{\hat{x}}{\underline{(\Sigma^K_{k=1}c_ku^k)}}||_2
		$$
		对于矩阵PCA而言，就相当于：
		$$
		\underset{Error}{\min}
		\left\{
		\begin{aligned}
		x^1-\bar{x}&\approx c_1^1u^1+c_2^1u^2+\dots \\
		x^2-\bar{x}&\approx c_1^2u^1+c_2^2u^2+\dots \\
		x^3-\bar{x}&\approx c_1^3u^1+c_2^3u^2+\dots \\
		&\dots
		\end{aligned}
		\right.
		\\
		\overset{m\times n}{X} \approx \overset{m\times k}{U} \quad \overset{k\times k} {\Sigma} \quad \overset{k\times n}{V}
		$$
		实际上，$U$的$k$列是一组正交的向量，他们是$XX^T$的特征向量，$k$个特征向量对应着$XX^T$中$k$个最大的特征值，而$XX^T$实际上就是$Cov(x)$，而用SVD解出的$U$实际上就是PCA得出来的解。
		
		因此，在训练中，$\hat{x}=\sum^K_{k=1}c_kw^k$，为了重建误差最小，$c_k=(x-\bar{x})\cdot w^k$，损失函数为公式(13)中的$L$。
		
		- 缺点：
		  1. 降维压缩特征导致特征丢失无法区分
		  2. 多维图形通过线性变换无法用有效的方法降维（比如将立体S形拉直，只能打扁）

## 总结

在本周的学习中，比较重点的关注了无监督学习中降维的数学原理，对最复杂的寻找投影的最大分布进行了一些推导。同时，也介绍了K-means及层次聚类两种比较基本的方法。下周将对非线性的无监督学习方法进行学习。
