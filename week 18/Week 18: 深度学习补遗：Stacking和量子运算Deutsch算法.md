# Week 18: 深度学习补遗：Stacking和量子运算Deutsch算法

## 摘要

本周在完成了李宏毅老师的ML Lecture 2017后，切换到了ML Lecture 2021进行学习，着重学习注意力机制等知识。同时继续推进量子计算相关学习内容。

## Abstract

After completing Professor Hung-yi Lee's ML Lecture 2017 this week, I switched to study ML Lecture 2021, focusing on topics like attention mechanisms. Concurrently, I continued advancing my learning in quantum computing-related content.

## 1. Stacking

Stacking的类似于对于$n$个分类系统$f_1\dots f_n$，输入$x$输入这$n$个分类系统中得到了$n$个输出$\gamma_1\dots\gamma_n$，最后输入一个训练的分类器中导出结果。

最后的这个Final Classifier不需要太复杂，使用简单的Logistics回归即可。 需要注意的是，需要将训练前面$n$个分类系统的数据集与训练Final Classifier的数据集分开，保证Final Classifier的权重准确性。

## 2. Self-Attention 自注意力

在传统、普通的机器学习应用中，输入一般是个定长的向量，但很多情况下（例如图、语音、文字等输入中），输入常常是一组不定长向量。通常的任务类型分为Sequence to Sequence、Sequence Labeling、Encoder-Only、Decoder-Only等等。

对于Sequence Labeling，即输入$n$个向量，一一对应的基于$n$个向量的输出。直观的想法是将每一个向量输入一个全连接层，对应一个输出。但对于例如一个英文句子“I saw a saw”而言，两个“saw”的语义和词性截然不同，但是对于一个全连接网络而言，其应该会输出相同的结果，但假如我们给两个“saw”加上不同的Tag，普通的全连接网络并不能学习这种知识。

这时候，就需要给网络中加入上下文信息，对于全连接网络而言，当然可以把前后词汇直接连接到下一个词汇的网络中，前后文的长度叫做Window，即窗口大小。但对于一个非常长的序列而言，如果需要整个序列的信息，就需要将整个序列之间全部链接才能得到全部的前后文，如果采用全连接网络，参数量会非常恐怖。

对于自注意力而言，其会将输入的所有向量都生成对应的输出向量，其中含有整个句子前后文的信息。这样，将嵌入全局信息之后的向量再对下一层的全连接层直接进行输入，全连接层就可以考虑全局的信息，而非仅仅考虑单个单词的信息。

![Self-attention with FC Layers](https://i-blog.csdnimg.cn/direct/5f736ac8ab7b44c79c413dfa4fce1c1b.png)

当然，自注意力层也可以进行嵌套，一层全连接层，一层自注意力层，使自注意力层专注于全局特征，全连接层专注于局部特征。

![Stacking Self-attention and FC Layers](https://i-blog.csdnimg.cn/direct/1ed1c408318a4281b5197558b3b4a735.png)

自注意力层的输入可以是整个模型的输入，也可以是其中的隐藏层的输入。已知自注意力层是考虑整个序列的输出结果，那么就需要求解两两输入之间的相关程度。求解两两之间的相关性有两种比较常用的方法，点积与加性方法。

点积就是将两个输入分别乘以矩阵$W^q$和$W^k$，得到两个结果$q$和$k$，对他们求点积$q\cdot k$，即逐元素的相乘相加，相关性就$\alpha=q\cdot k$。

加性的计算方法不同之处在于，经过变换求得$q$与$k$后，将其相加并经过一个激活函数（例如$tanh$）后再经过一次变换$W$后得到相关系数$\alpha$。

如今最常用的，也是被用在Transformer中的求解相关性的方法是点积方法。

![Self-attention Structure](https://i-blog.csdnimg.cn/direct/9f812519aff14bbaaacf53fed1c5aed3.png)

对于主元$a^1$，我们将其称为“Query”，要对每一个元素（包括其自身）$a^t$（称为“Key”）求解相关性。而每一个$q^t$、$k^t$都是由对应的元素乘以对应的变换得到。紧接着，对当前的$q^i$和每一个$k^j$求点积，得到一个注意力分数$a_{i,j}$。再把$\left.a_{i,j}\right|_{i=t}$经过Softmax后得到一个激活的注意力分数$a'_{i,j}$（可以换用别的激活函数）。
$$
\begin{aligned}
q^1 &= W^q a^1\\
k^1 &= W^k a^1\\
k^2 &= W^k a^2\\
&\dots \\
a_{1,1}&=q^1\cdot k^1\\
a_{1,2}&=q^1\cdot k^2\\
&\dots \\
\end{aligned}
$$

![Self-attention Calculation](https://i-blog.csdnimg.cn/direct/06f9fb1852b741b6aef0d7ebab35f2cd.png)

接着，将每个$a^i$乘以一个矩阵$W^v$得到对应的$v^i$，即$v^i=W^va^i$，再经过式子$b^1=\sum_ia'_{1,i}v^i$即可得到$a^1$对应的输出。

对于每个$a^i$，求出其对应的$b^i$，这样就完成了自注意力的求解。

## 3. 量子计算 Deutsch算法

Deutsch算法可以这么被描述。
$$
(H \otimes I) \, U_f \, (H \otimes H) \, \ket{0, 1}
$$
即将$\ket{0,1}$经过两个$Hadamard$门后，进入$U_f$门，最后对两个输出一个输入$Hadamard$门，一个输入单位门。

- 将$\ket{0,1}$，即$\ket{0}\otimes\ket{1}$输入$Hadamard$门，$ H\ket{0} \otimes H\ket{1} = \frac{1}{\sqrt{2}} (\ket{0} + \ket{1}) \otimes \frac{1}{\sqrt{2}} (\ket{0} - \ket{1}) $。
- $U_f$定义为$\ket{x, y} \mapsto \ket{x,y \oplus f(x)} $，那么对于$\ket{\frac{1}{\sqrt{2}} (\ket{0} + \ket{1}), \frac{1}{\sqrt{2}} (\ket{0} - \ket{1})}$的输入，$x$为控制比特，$y$为目标比特。即实际计算中，$\ket{\frac{1}{\sqrt{2}} (\ket{0} + \ket{1}), \frac{1}{\sqrt{2}} (\ket{0} - \ket{1})}=\ket{x,\frac{\ket{0}\oplus f(x)-\ket{1}\oplus f(x)}{\sqrt{2}}}$
- 又因为$x=\frac{1}{\sqrt{2}} (\ket{0} + \ket{1})$，由于量子计算的叠加性，实际上可以进行拆分，因此可以对$\ket{0}$和$\ket{1}$分开讨论。若$f(x)=0$，那么$0\oplus0=0$，$1\oplus0=1$，没有变化。若$f(x)=1$，那么$0\oplus1=1$，$1\oplus1=0$，完成翻转。也就是$\frac{\ket{0}-\ket{1}}{\sqrt{2}}$会被翻转为$\frac{\ket{1}-\ket{0}}{\sqrt{2}}=-\frac{\ket{0}-\ket{1}}{\sqrt{2}}$
- 因此，如果$f(0)=f(1)$，那么$\ket{0}$和$\ket{1}$同号，即在$y=x$上；如果$f(0)\neq f(1)$，$\ket{0}$和$\ket{1}$异号，就在$y=-x$上。在经过一个$Hadamard$门后，分别应该被反射到$x=0$和$y=0$上，即$\ket{0}$和$\ket{1}$。

因此，只需要经过一次测量，就可以测得是否$f(1)=f(0)$。

## 总结

本周学习了Stacking的集成学习方法，完成了ML Lecture 2017的学习。同时开始了ML Lecture 2021的学习，学习了自注意力相关底层知识，了解了自注意力层的整个运算步骤和原理，预计下周继续学习自注意力，对其底层逻辑继续进行深入研究。本周还进行了量子运算学习的推进，主要学习了Deutsch算法，了解了对死/活黑箱的探测原理，和其与传统运算的一些区别。