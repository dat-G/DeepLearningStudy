[TOC]

# Week 8: 深度学习补遗：词嵌入

## 摘要

本周对李宏毅老师的课程继续进行学习，对无监督学习继续进行学习，了解了通过矩阵分解来预测的思想以及在自然语言处理中非常重要的词嵌入及词袋模型等概念，同时对邻近嵌入进行了一定的了解。

## Abstract

This week's study of Professor Hung-yi Lee's course continues with unsupervised learning, the idea of prediction through matrix decomposition and the concepts of word embedding and bag-of-words model, which are very important in natural language processing, as well as a certain understanding of neighbor embedding.

## 1. 矩阵分解 Matrix Factorization

对于一个离散的二维分布而言，可以设想其每个元素与其横纵轴对应元素存在一定的联系，矩阵分解的目的即为找到他们的潜在联系。

比如下图中，对于一个代表每个人持有手办数量关系的矩阵$X$，其各纵列为各个动画人物，各橫行为各人。

![Matrix X](https://i-blog.csdnimg.cn/direct/7c22f2a9191f49f6b99e55015dc0da1d.png)

可以假设存在关系：
$$
r^A\cdot r^1\approx5 \\
r^B\cdot r^1\approx4 \\
r^C\cdot r^1\approx1 \\
\vdots
$$
因此可以考虑分解：
$$
\sideset{\small{M}}{\overset{N}{
\begin{bmatrix}
n_{A1}&n_{A2}&\dots \\
n_{B1}&n_{B2}&\dots \\
\vdots&\vdots
\end{bmatrix}}}{}
\underset{\text{Minimize Error}}{\approx}
\sideset{\small{M}}{\overset{K}{
\begin{bmatrix}
r^A\\
r^B\\
\vdots
\end{bmatrix}}}{}
\times
\sideset{\small{K}}{\overset{N}{
\begin{bmatrix}
r^1&r^2&\dots
\end{bmatrix}}}{}
$$
假设矩阵$X$有一些元素未知，如下：

![Masked Matrix X](https://i-blog.csdnimg.cn/direct/98f1f67c839c4a1083b90fa51ed7fad1.png)

就可以得出一个损失函数：$L=\sum_{(i,j)}(r^i\cdot r^j-n_{ij})^2$，因此可以用梯度下降法拟合出$r^i$和$r^j$。

而有时关系并不是简单的积的关系，因此也可以对模型进行修改：$r^A\cdot r^1\approx5 \to r^A\cdot r^1+b_A+b_1\approx5$给模型加上人和角色相应的偏差值，其损失函数也会变化为$L=\sum_{(i,j)}(r^i\cdot r^j+b_i+b_j-n_{ij})^2$。

而矩阵分解的技术在主题分析上的运用比较经典的有LSA（潜在语义分析，Latent Samantic Analysis），即把上述中的矩阵的人换成关键词，动漫人物换成文章，持有手办数量换成词语出现次数，就可以达到相应的目的。

## 2. 词嵌入 Word Embedding

在自然语言处理领域中，我们常常想要将自然语言的词汇向量化，最容易想到的是所谓的1-of-N Encoding嵌入法。
$$
\text{1-of-N Encoding} \\
\begin{aligned}
apple&=\begin{bmatrix}1&0&0&0&0\end{bmatrix} \\
bag&=\begin{bmatrix}0&1&0&0&0\end{bmatrix} \\
cat&=\begin{bmatrix}0&0&1&0&0\end{bmatrix} \\
dog&=\begin{bmatrix}0&0&0&1&0\end{bmatrix} \\
elephant&=\begin{bmatrix}0&0&0&0&1\end{bmatrix}
\end{aligned}
$$
但是这样的维度会过大，对于一个词语库，可能会达到几万维。同时，这样的向量化表达无法获取两个词语之间的临近关系，能获取的信息也有限。于是，可以考虑用词语之间的类别关系，把词语映射到高维空间。

![Word Embedding](https://i-blog.csdnimg.cn/direct/9c6b93454a7140518401bf21d39c1243.png)

通过这样映射，可以看到，同属于一个类别的词语距离会相对比较近。而词嵌入的工作由机器大规模的进行无监督学习生成词向量，而机器之所以能无监督地明白词汇的含义，是通过了大量的前后文阅读来了解同一个词语的意思。

- 基于计数的方法 Count Based

  如果两个单词$w_i$和$w_j$经常一起出现，向量$V(w_i)$和$V(w_j)$的应该靠近。一个很经典的例子是Glove Vector，假设$w_i$和$w_j$在同一片文档出现的次数为$N_{i,j}$，应该使向量的内积$V(w_i)\cdot V(w_j)\leftrightarrow N_{i,j}$。

- 基于预测的方法 Prediction Based

  训练一个神经网络，通过前文预测后文的内容，就可以进行无监督学习。
  $$
  \dots\quad w_{i-2}\quad w_{i-1}\underset{Predict}{\to}\_\_\_
  $$
  实际上操作时，是将$w_{i-1}$进行1-of-N Endcoding进行输入，每一个维度代表那个词出现的概率，而输出也是一个1-of-N Endcoding输出，代表$w_i$的各单词可能性，将第一个隐藏层称为$z$。则在训练完成的模型中，每一个单词预测所得到的$z^i$向量就是这个词的Word Embedding。

  例如“我吃饭”和“小明吃饭”中，在训练过程中，“我”和“小明”最终都需要预测出“吃饭”的结果，在训练的梯度更新过程中就会让这两个词汇的$z$在向量空间上更加接近，利用这个特性，从而完成了词嵌入的流程。

  也可以将$w_{i-1}$和$w_{i-2}$都输入神经网络参与训练，这个时候应该让其相同位置的元素共享参数。需要注意$w_{i-1}$和$w_{i-2}$的长度都应该是$|V|$，而$z$的长度不一定相同，记作$|Z|$。有$\vec{z}=\vec{W_1} \vec{x_{i-2}}+ \vec{W_2}\vec{x_{i-1}}$，而因为共享权重的原因，强制$W_1=W_2=W\to \vec{z}=\vec{W}(\vec{x_{i-2}}+\vec{x_{i-1}})$。
  
  最后，利用交叉熵损失评估预测并进行反向传播优化神经网络。这种方法有一系列变种，例如连续词袋模型（CBOW，Continuous Bag of Word Model），$\dots w_{i-1}\underset{Predict}{\to}\_\_\_ \underset{Predict}{\gets}w_{i+1}\dots$，通过前后文同时输入来预测预测文字进行训练；还有Skip-gram模型，$\_\_\_\underset{Predict}{\gets}W_i\underset{Predict}{\to}\_\_\_$，通过中间词汇预测前后词汇。
  
  存在一个比较重要的Trick，就是一般这种神经网络不需要过于deep，会降低很多的效率而并不能取得多少收益。

![Word Embedding Example](https://i-blog.csdnimg.cn/direct/4150a06e32e8480a8fa29077d80e8d62.png)

认为$V(hotter)$是$hotter$经过Embedding后的词向量，有下列特性存在：
$$
\begin{aligned}
V(hotter)-V(hot)&\approx V(bigger)-V(big) \\
V(Rome)-V(Italy)&\approx V(Berlin)-V(Germany) \\
V(king)-V(queen)&\approx V(uncle)-V(aunt)
\end{aligned}
$$
 根据以上性质就可以对类比进行求解：
$$
\text{Question:}Rome:Italy=Berlin:? \\
\underset{\text{Find the word }w\text{ with the closest }V(w)}{\underline{V(Berlin)-V(Rome)+V(Italy)}}\approx V(Germany)
$$
但词袋模型有其巨大的局限性，最简单的例子就是，两个英文句子：

- White blood cells destroying an infection.
- An infection destroying white blood cells.

两个句子虽然具有相同的词汇，但顺序的排布带来截然不同的意思，因此光使用词袋模型是完全不足够的。

## 3.近邻嵌入 Neighbor Embedding

欧式距离在曲面上会被扭曲，经典的例子是在地球上，只有在很短的两点上，欧氏距离才会误差比较小，因为地球是个球体，而经纬度坐标实际上是球面坐标，需要转化为平面坐标（高斯坐标）才能直接使用欧式距离。而在高维向量空间上也常常遇到这种情况。

流式学习 Manifold Learning 旨在将高维空间展平，用于解决上述的问题，本质上是一种降维操作。如下图，流式学习可以将左图的S形平面展平成右图，这样可以更好地衡量两点之间的距离。

![Manifold Learning](https://i-blog.csdnimg.cn/direct/3a5449a35884484794aeb1b5ffff6663.png)

### 3.1 局部线性嵌入 Locally Linear Embedding

![Locally Linear Embedding](https://i-blog.csdnimg.cn/direct/d1c49d518d8f4bd28e07a49759f813eb.png)

局部线性嵌入也是一种经典降维方法，即对于每个样本点$x^i$，先选择$K$个近邻$x^j$进行线性组合，得到$x^i\approx w_{ij}x^j$，于是需要求$arg\underset{w_{ij}}{\min}\sum_j||x^i-w_{ij}x^j||_2$。找到一组$w_{ij}$后保持$w_{ij}$不变，用$z^i$和$z^j$替代$x^i$、$x_j$，求解$arg\underset{z^i,z^j}{\min}\sum_j||z^i-w_{ij}z^j||_2$，便可以将$x^i$、$x_j$降维到$z^i$和$z^j$上，同时保持$x^i$与$K$个近邻$x^j$之间的相对关系，但$K$的选择应该适当，否则就会让降维的效果变差，保留过多的和较远点之间的信息。

![K's impact in LLE's Variance](https://i-blog.csdnimg.cn/direct/4868b727d63f488ab071eb9d4c7f182f.png)

### 3.2 拉普拉斯映射 Laplacian Eigenmaps

在半监督学习中提及了平滑正则化，即两个点的欧式距离并不能完全反应其相似性（真正的“距离），而其”在高密度区域下接近才是真正的接近。可以基于样本点建立一个图，两点之间的平滑距离可以用两个点连接路径中的边数来近似表示。

有平滑假设 Smothness Assumption：如果高密度区域中两个样本点$x^i$、$x^j$是相近的，他们的标签$y^i$和$y^j$很可能相同，则训练时的损失函数如下。
$$
\begin{aligned}
L&=\sum_{x^r}C(y^r,\hat{y^r})+\lambda S \\
S&=\frac{1}{2}\sum_{i,j}w_{i,j}(y^i-y^j)^2=\bold{y}^TL\bold{y}
\end{aligned}
$$
$C(y^r,\hat{y^r})$是有标签数据的损失，$\lambda S$是无标签数据的损失（作为一个正则项，根据当前标签的平滑程度做惩罚）。如果$x^i$和$x^j$是相连的，$w_{i,j}$为两点的相似度、否则为0。因此$S$的含义为，如果$x^i$与$x^j$很接近，那么$w_{i,j}$就会有一个很大的值，就会希望$y^i$、$y^j$越接近越好。

而在无监督学习中仅仅要求$S=\frac{1}{2}\sum_{i,j}w_{i,j}(z^i-z^j)^2$最小，是行不通的，因为如果所有$z^i$、$z_j$都相同，那么$S$就会等于0，是无意义的。需要增加一些限制，假如降维后$z$处于$M$维空间，希望$z$占据整个$M$维空间，而不是比$M$维更小的空间。这个式子解得的$z$其实就是图拉普拉斯矩阵$L$比较小的特征值对应的特征向量，如果通过拉普拉斯特征映射找到$z$后再利用K-means做聚类分析，就叫做谱聚类 Spectral Clustering。

## 总结

本周初步了解了词嵌入的相关知识，对自然语言中的词语如何映射成向量有了初步的认识。而临近嵌入等映射方法又对降维操作提供了另一种不同的思路和启发，主要是对于高维空间曲面的有效展平进行了研究，对比只关注映射后变化幅度的PCA方法，近邻映射更加关注点之间的邻近关系，对于含有隐含关系的点云处理有比较重要的意义。预计下周将学习Auto-Encoder相关内容。