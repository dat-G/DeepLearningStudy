[TOC]

# Week 9: 深度学习补遗：自编码器与生成式模型

## 摘要

本周，跟随李宏毅老师的课程学习了自编码器和生成式模型的课程，两方面的内容关系比较紧密。主要从抽象角度以及一些数学角度对自编码器进行了学习，编码器-解码器架构作为现在模型的一大主流结构，具有比较重要的学习意义，对自编码器以及生成式模型的学习使我对编码器-解码器架构有了一定了解。

## Abstract

This week, I followed Mr. Hung-yi Lee's course to learn about self-encoders and generative models, which are closely related to each other. Mainly from an abstract point of view as well as some mathematical point of view on the self-encoder learning, encoder-decoder architecture as a major mainstream structure of the model now, has a more important learning significance, on the self-encoder as well as the generative model of the learning of the encoder-decoder architecture so that I have a certain understanding of the encoder-decoder architecture.

## 1. t分布随机邻居嵌入 t-Distributed Stochastic Neighbor Embedding

LLE与拉普拉斯映射只考虑了“临近点必须接近”，并没有考虑“非临近点要远离”。所以实际上，LLE在会把同一个类别的点聚在一起的同时，难以避免不同的类别的点重叠在一起。导致类别之间难以区分。

而t-SNE的要点是，在原空间内计算每一对点$x^i$与$x^j$之间的相似度$S(x^i,x^j)$，然后做归一化$P(x^j|x^i)=\frac{S(x^i,x^j)}{\sum_{x\neq k}S(x^i,x^k)}$。在降维之后获取到的点$z^i$、$z^j$上同样也计算相似度$S'(z^i,z^j)$，并归一化$Q(z^j|z^i)=\frac{S'(z^i,z^j)}{\sum_{x\neq k}S'(z^i,z^k)}$。（归一化的主要目的是，统一不同维度下相似度的scale）。

我们希望两个分布的相似度分布越接近越好，因此可以用KL散度衡量两个分布的相似度分布的接近程度。
$$
\begin{aligned}
\text{minimize}L&=\sum_i KL(P(*|x^i)||Q(*|z^i)) \\
 &=\sum_i \sum_j P(x^j,x^i) log\frac{P(x^j,x^i)}{Q(z^j,z^i)}
\end{aligned}
$$
t-SNE的计算量比较大，对于一个比较高维度的一个向量，可能会造成比较大的运算压力。一个常见的方法是先对

用PCA之类的方法进行降维（比如从200维降到50维度），再用t-SNE完成降维（从50维到2维）。t-SNE对新出现的样本点，需要重新训练整个模型，因此常常被用在可视化上而非模型训练上。

在t-SNE之前，有SNE算法，其衡量相似度的算法是使用RBF function，即$S'(z^i, z^j)=e^{-||z^i-z^j||_2}$，而t-SNE在降维后使用t分布$S'(z^i, z^j)=\frac{1}{1+||z^i-z^j||_2}$计算两个样本的相似度。在可视化上，可以看到t-SNE随着迭代，不同类别样本被分得越来越开。

![t-SNE on MNIST](https://i-blog.csdnimg.cn/direct/9a5debb9a5c94a889c229156b4b259a8.png)

## 2. 自编码器 Auto-Encoder

单独一个编码器或一个解码器对半监督学习可用，但对于无监督学习来说不可用，因为单编码器或单解码器必须要有带标注的数据（NN-Encoder或NN-Decoder）进行训练。对于无监督学习，自编码器是一个好的选择，所谓自编码器，就是在一个神经网络内同时放入编码器和解码器，就可以在无监督学习的情况下进行自训练。

输入层到隐藏层的步骤也被称为编码（“Encode”），隐藏层到输出层的步骤也被称为解码（“Decode”），隐藏层也被称为“瓶颈层”（“Bottleneck Layer”）。

![Auto-Encoder](https://i-blog.csdnimg.cn/direct/3a98622642d84429b8b85b758f0c8a23.png)

自编码器模型也可以变得非常的Deep，在一个Deep Auto-Encoder中，总会中间有一个层特别窄，我们称其为瓶颈层，瓶颈层神经元的输出就被称为Code。而输入层到瓶颈层之间都被称为Encoder，瓶颈层到输出层之间都被称为Decoder。

- 在文本检索中，可以把每一个文档和每一个查询都表达为一个向量，计算向量空间内所有文档和查询的相似度（运用内积或者余弦函数），最相似的那个作为检索的结果。将文档和查询映射到向量的最简单方法是词袋模型，但词袋模型无法捕捉语义，而Deep Auto-Encoder可以综合考虑语义，因此更优。

- 在图片检索中，基于像素级直接输入模型计算相似度得到的结果并不好，但通过Deep Auto-Encoder先对图片进行编码再去查找计算相似度，就可以大大提高结果。又因为Deep Auto-Encoder是基于无监督学习的，所以并不缺乏数据，可以用广泛的数据对其进行训练。

去噪声自编码器，则是通过在输入层后加入随机噪声后再进入编码层，让输出层尽可能接近不带噪声的输入图像，这样的训练方式可以增强模型的鲁棒性。实现增强模型鲁棒性的另一个方式是收缩自编码器，通过最小化输入变化带来的Code变化，让噪声对编码结果的影响最小，也可以增强鲁棒性。

## 3. CNN上的Auto-Encoder

可以想象，如果需要对CNN做自编码器，解码部分就应该和编码部分相反，也就是需要出现反池化与反卷积。反池化步骤比较容易理解，有两种比较主流的实现方法，一个是将元素放回原来处于的位置，剩下置0；另一个是将四个格子都填上同一个数值。

![Unpooling](https://i-blog.csdnimg.cn/direct/ddeba74bbb6345f5ae5f1d78e211fa66.png)

反卷积相对比较复杂，但简单来说，卷积的步骤是将多个值变为一个值，反卷积就是要将一个值拓展为多个值，重复的部分进行相加。实际上，反卷积也是一种卷积。

![Deconvolution](https://i-blog.csdnimg.cn/direct/f6891eba05d745d483f9b7d0ef0a6583.png)



## 4. 生成式模型 - PixelRNN

Pixel RNN完全是一个无监督的生成式模型，通过输入前一个像素，预测后一个像素来生成整个图像，虽然它非常简单，但其确实奏效。

![Pixel RNN](https://i-blog.csdnimg.cn/direct/b94528a8aac6462ea4d6f4f14ce81503.png)

但其并不能很好的理解图片，会出现把狗的下半身绘制成鸡、猴子之类的情况，但图片基本上是流畅的。

## 5. Variational Auto-Encoder - 变分自编码器

VAE对比Auto-Encoder来说使用了一个trick，就是例如想要将输入编码成三维的，就引出两个三维向量$m$和$\sigma$，同时从正态分布生成一个三维向量$e$，让下一层$c$的每一个$c_i=\exp(\sigma_i)\times e_i+m_i$。同时最小化重建损失，即有：
$$
arg\underset{\sigma_i,m_i}{\min}\sum_{i=1}^3(1+\sigma_i-(m_i)^2-\exp(\sigma_i))
$$


![VAE](https://i-blog.csdnimg.cn/direct/a5caa24e2f3c4fa589f8037d32436628.png)

VAE对比PixelRNN的主要优势，其实是可以控制生成的参数。训练好模型后，提取NN Decoder部分，生成的输入部分就是向量$\bold c$，这时控制输入向量的变化就可以让生成的图片产生不同的变化。

## 总结

本周继续对无监督学习的探索，利用词嵌入的知识，继续对自编码器进行了学习。理解到了自编码器的巧思，利用无标注数据进行无监督学习，解决了数据标注的问题，同时也革新了生成式模型的边界。下周计划继续完成深度生成式网络的学习，在这两周进入支持向量机的学习。

