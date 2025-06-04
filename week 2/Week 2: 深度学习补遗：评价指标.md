[TOC]

# Week 2: 深度学习补遗：评价指标

## 绪论

本周修改了一下目标，将对一些理论方面进行夯实，对评价指标的数学理论以及实现方面进行探索。

## 1 深度学习

### 1.1 深度学习任务目标

- **分类（Classification）**：分类任务的目标是根据输入的特征，将其分配到预定义的离散的类别中。可以是二分类（“是”、“否”），也可以是多分类（“苹果”、“香蕉”、“橙子”等）。输出离散的、有限的标签，例如0\~9的手写数字识别，通常将输出10个在正负无穷开区间的浮点数，越大代表概率越高。因此可以用Softmax函数将其映射到$(0,1)$之间变成概率。Softmax的表达式为$\begin{equation*} Softmax(z_i)= \frac {e^{z_i}} {\sum_{c=1}^C{e^{z_c}}}  \end{equation*}$，定义域为$(-\infty,+\infty)$，值域为$(0, 1)$。
- **回归（Regression）**：回归任务的目标是根据输入特征找到一个连续数值，通常是一个特定范围内的浮点数。模型在回归任务中学习的是输入特征与输出特征之间的关系，以此做出准确的预测。典型的任务有预测房屋价格、天气温度或股票价格。

### 1.2 深度学习评价指标

#### 1.2.1 分类任务评价指标

二分类任务下，预测结果将分为以下几类。

| 真实 \ 预测 | 正                 | 反                 |
| ----------- | ------------------ | ------------------ |
| 正          | True Positive, TP  | False Positive, FP |
| 反          | False Negative, FN | True Nagative, TN  |

可见，后一位字母为预测值，预测为正则为P，预测为反则为N。而预测结果与真实值相同时前一位为T，预测结果不符时为F。

而多分类任务下，也同理，可将当前类别认为是正类，其余类别均认为为反类，也可以计算下述指标。

##### 1.2.1.1 Accuracy 准确率

$$
\begin {equation}
Accurancy = \frac {TP + TN} {TP + FP + TN + FN}
\end {equation}
$$

准确率，表示**正确预测数量占总数的比例**，他是最直观的评价指标之一。但在类别不平衡的情况下，准确率可能会给出误导性的结果。例如，在一个疾病诊断任务中，健康样本占比 99%，疾病样本占比 1%，即使模型将所有样本都预测为健康，准确率也能达到 99%，但实际上模型对疾病样本的预测能力为零。

##### 1.2.1.2 Precision 精确率

$$
\begin{equation}
Precision = \frac {TP} {TP+FP}
\end{equation}
$$

精确率也称为查准率，是指模型**预测为正类的样本中，真正为正类的样本比例**，精确率关注的是模型预测为正类的结果中有多少是正确的。在一些对误判正类有较高成本的场景中，如垃圾邮件分类，精确率尤为重要，因为将正常邮件误判为垃圾邮件会给用户带来较大的困扰。

##### 1.2.1.3 Recall 召回率

$$
\begin {equation}
Recall = \frac {TP}{TP+FN}
\end {equation}
$$

召回率也称为查全率，是指**真实正类样本中，被模型正确预测为正类的样本比例**。召回率衡量了模型找到所有正类样本的能力。在一些需要尽可能找到所有正类样本的场景中，如疾病筛查，召回率非常关键，因为漏诊一个患病患者可能会导致严重的后果。

##### 1.2.1.4 F1 Score

$$
\begin{equation}
\text{F1 Score} = \frac {2\times\text{Precision}\times\text{Recall}} {\text{Precision}+\text{Recall}}
\end{equation}
$$

F1 值是精确率和召回率的调和平均数，**F1 值综合考虑了精确率和召回率**，当精确率和召回率都较高时，F1 值也会较高。它在需要同时平衡精确率和召回率的场景中非常有用，能够更全面地评价模型的性能。

##### 1.2.1.5 ROC Curve（Receiver Operating Characteristic Curve）

ROC曲线是以假正例率（False Positive Rate, FPR）为横轴，真正例率（True Positive Rate, TPR）为纵轴绘制的曲线。
$$
\begin{equation}
FPR = \frac{FP}{FP+TN}
\end{equation}
$$

$$
\begin{equation}
TPR=\frac{TP}{TP+FN}=Recall
\end{equation}
$$

可知，假正例率FPR真实标签为假的样本中被预测为真的样本的比例，而真正利率等于召回率。

将其绘图后，可以认为**一个好的模型的ROC曲线应该尽可能靠近左上角，即假正例率低、真正例率高**。

![ROC curve](https://i-blog.csdnimg.cn/direct/2a650906653b49e292f54f33e0e5a4a7.png#pic_center)

> 图片来源：[Wikipedia：ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#/media/File:Roc_curve.svg)

##### 1.2.1.6 AUC（Area Under the Curve）

AUC值是ROC曲线下的面积，易知，当AUC值为0.5时，即相当于上图中的Random Classifier，即为随机分配，当AUC值越接近1时，模型效果越好。

AUC的特点是，**不受不平衡类别的影响**，因此在处理不平衡数据集时非常有用。

> 参考文章：[CSDN：深度学习常用参数&评估指标详细汇总](https://blog.csdn.net/bkirito/article/details/145774610)

#### 1.2.2 回归任务评价指标

##### 1.2.2.1 MSE 均方误差

$$
\begin{equation}
MSE = \frac{1}{n}\sum^{n}_{i=1}(y_i-\hat{y_i})^2
\end{equation}
$$

均方误差主要计算的是真实值与预测值之间差值的平方的平均值，由于MSE进行了平方运算，异常值会被放大，因此，MSE是一个对异常值十分敏感的指标。在实际应用中，比如在预测股票价格走势时，如果某一天股票价格因为突发的重大事件而出现异常波动，那么这个样本的误差在 MSE 的计算中会被显著放大，进而对整体的评估结果产生较大影响。

##### 1.2.2.2 MAE 平均绝对误差

$$
\begin{equation}

MAE=\frac{1}{n}\sum^n_{i=1}\left|y_i-\hat{y_i}\right|
\end{equation}
$$

平均绝对误差计算的是真实值与预测纸质件差值的绝对值的平均值，其对异常值较不敏感，在一些不需要突出异常值的评估场景中比较适用。

#### 1.2.3 目标检测 / 分割任务评价指标

##### 1.2.3.1 IoU 交并比

$$
\begin {equation}
IoU = \frac {预测区域与真实区域的交集面积} {预测区域与真实区域的并集面积}
\end {equation}
$$

IoU（Intersection over Union）即交并比，是目标检测和图像分割任务中常用的评价指标。它通过计算预测区域（如目标检测中的预测框、图像分割中的预测掩码）与真实区域（真实标注的目标框或掩码）之间的交集面积与并集面积的比值，来衡量两者的重合程度。如果预测框与标注框完全重合，则$IOU=1$。

##### 1.2.3.2 mAP 全类平均正确率

 对于每个类别，通过改变检测阈值（如分类器的置信度阈值），可以得到一系列的精度和召回率值，从而绘制出精度 - 召回率曲线。该**曲线下的面积**就是该类别的平均精度（Average Precision，AP）。**mAP则是对所有类别的AP值求平均**，得到的平均值。

我们一般会看到  mAP（平均精度均值）出现 0.5 和 0.9 这样的取值，一般写成 mAP@0.5 和 mAP@0.9 或 mAP@0.5:0.95 等形式，这主要与评估时所采用的交并比（IoU）阈值有关。

例如mAP@0.5表示在计算 mAP 时，以 0.5 作为 IoU 的阈值来判断检测结果是否正确。即只有当预测框与真实框的 IoU 大于等于 0.5 时，才认为该检测是一个正确的预测，否则视为错误预测。

### 1.3 代码实现与理论补充

在Week 1[[Github](https://github.com/dat-G/DeepLearningStudy/blob/main/week%201/Week%201:%20Python%E7%B1%BB%E4%B8%8E%E7%BB%A7%E6%89%BF%E8%A1%A5%E9%81%97%E4%B8%8EPyTorch%E5%85%A5%E9%97%A8.md) / [CSDN](https://blog.csdn.net/MCHacker/article/details/148365489)]的MNIST项目后，增加评估模块。

```python
model.eval()
total_correct = 0
total = 0

num_classes = 10 # MNIST有10个类别 (0-9)
metrics_per_class = {i: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for i in range(num_classes)}

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        output = model(images)
        _, predicted_class = torch.max(output.data, 1)

        for class_id in range(num_classes):
            # 将当前 class_id 视为“正类”
            is_positive_class = (labels == class_id)
            is_negative_class = (labels != class_id)

            # 模型预测为当前正类
            predicted_positive = (predicted_class == class_id)
            predicted_negative = (predicted_class != class_id)

            metrics_per_class[class_id]['TP'] += (predicted_positive & is_positive_class).sum().item()
            metrics_per_class[class_id]['FP'] += (predicted_positive & is_negative_class).sum().item()
            metrics_per_class[class_id]['TN'] += (predicted_negative & is_negative_class).sum().item()
            metrics_per_class[class_id]['FN'] += (predicted_negative & is_positive_class).sum().item()

        total += labels.size(0)
        total_correct += (predicted_class == labels).sum().item()
print(f'Accuracy: {100 * total_correct / total:.2f}%')
```

对于多分类问题，应该分类计算，每一类计算时将自己视为正类，将其他类视为负类。

Pytorch进行两个张量的逻辑运算时，会将对应的位置元素进行逻辑运算后在对应位置留下布尔计算结果（0或1），因此执行`.sum().item()`后可以统计当前张量中成真的元素的数量。

需要补充几个附加知识以正确的计算这些参数。

#### 1.3.1 宏平均（Macro-average）和微平均（Micro-average）

> 参考文章：[博客园：宏平均和微平均](https://www.cnblogs.com/sddai/p/15174560.html)

##### 1.3.1.1 宏平均

独立计算每一个类别的指标（精确率 Precision、召回率 Recall、F1 Score），再求平均。

以精确率 Precision为例：
$$
\begin {equation}
\text{macro-Precision}=\frac{1}{n}\sum^n_{i=1}Precision_i=\overline{Precision}
\end {equation}
$$
宏平均平等的对待每个类别，但容易受到数量不平均的类别的影响。

```python
macro_precision = sum(tp / (tp + fp) for tp, fp, _, _ in [(m['TP'], m['FP'], m['TN'], m['FN']) for m in metrics_per_class.values()] if (tp + fp) > 0) / num_classes
macro_recall = sum(tp / (tp + fn) for tp, _, _, fn in [(m['TP'], m['FP'], m['TN'], m['FN']) for m in metrics_per_class.values()] if (tp + fn) > 0) / num_classes
macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0
```

##### 1.3.1.2 微平均

汇总所有的真正例（TP）、假正例（FP）、假负例（FN）后计算全局指标。

以精确率 Precision为例：
$$
micro-Precision=\frac{1}{n}\sum^n_{i=1}\frac{TP_i}{TP_i+FP_i}=\frac{\overline{TP}}{\overline{TP}+\overline{FP}}
$$
微平均平等的对待每个样本，适合样本数量差距较大的数据不均衡场景。

```python
total_tp = sum(m['TP'] for m in metrics_per_class.values())
total_fp = sum(m['FP'] for m in metrics_per_class.values())
total_fn = sum(m['FN'] for m in metrics_per_class.values())

precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
```

#### 1.3.2 多样本情况下的准确率计算问题

二分类的准确率计算公式为$Accurancy = \frac {TP + TN} {TP + FP + TN + FN}$，容易得出代码`total_tn = sum(m['TN'] for m in metrics_per_class.values()) `用于计算宏平均准确率。但实际上，这并不是准确的多分类准确率计算方法。

**在每个样本进行预测时，每个正确预测，会给其预测类别增加一个TP，但也会给其他所有类别增加一个TN。**同理每一次错误分类（例如，真实标签是 A，预测为 B）都会为类别 A 贡献一个 FN，同时为类别 B 贡献一个 FP。因此，所有类别的 FP 总数必然等于所有类别的 FN 总数。

因此需要在评估时计算`total_correct`和`total`，直接用`total_correct / total`计算准确率。

#### 1.3.3 MNIST FCNN模型评估结果

利用微平均评估的结果如下。

```bash
100%|██████████| 157/157 [00:00<00:00, 754.23it/s]
Accuracy: 97.39%
Precision: 0.9739
Recall: 0.9736
F1-Score: 0.9738
```

## 总结

本周对各个评估指标的计算方式进行了探索和夯实，针对上周的FCNN实现的MNIST手写识别任务简单编写了评估模块，同时对在编写方面发现的宏平均和微平均问题方面进行了探索。下周暂定探索优化器和损失函数，后续将会由浅入深对各种经典的网络结构进行学习。