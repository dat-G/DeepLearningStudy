[TOC]

# Week 22:

## 摘要

## Abstract

## 1. Positional Encoding 位置编码

$$
PE_{(pos,2i)}=\sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}}) \\
PE_{(pos,2i+1)}=\cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}}) \\
$$

Transformer的位置编码对于Token中奇偶位置的数字采取不同的位置编码。

这样设计的目的主要是由于三角函数的和差角公式的存在。
$$
\left\{
\begin{aligned}
\sin(\alpha+\beta)&=\sin\alpha\cos\beta+\cos\alpha\sin\beta \\
\cos(\alpha+\beta)&=\cos\alpha\cos\beta-\sin\alpha\sin\beta
\end{aligned}
\right.
$$
可以得到，
$$
PE_{(pos+k,2i)}=PE_{(pos,2i)}\times PE_{(k,2i+1)}+PE_{(pos,2i+1)}\times PE_{(k,2i)}\\
PE_{(pos+k,2i+1)}=PE_{(pos,2i+1)}\times PE_{(k,2i+1)}-PE_{(pos,2i)}\times PE_{(k,2i)}
$$


## 总结