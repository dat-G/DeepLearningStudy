[TOC]

# Week 1: Python类与继承补遗与PyTorch入门

## 绪论

近几周学习专注于深度学习基础知识的补齐和理论能力与代码能力的对齐，将本科毕业设计和课题组中学到的一些名词和理论进行夯实。

##  1. PyTorch

PyTorch是一款主要由Facebook的团队开发的深度学习框架，基于Python语言。对比TensorFlow，其易用性、灵活性、高效性比较突出，具有几个主要特点。

- **动态计算图**（还不太了解）：TensorFlow主要采用静态图机制，PyTorch采用运行时定义计算图的方式，便于调试开发复杂模型。
- **自动求导**：强大的Autograd模块，可以自动计算梯度，极大简化了反向传播算法实现的流程。
- **丰富的API**：提供了张量运算（`tensor`类型）、神经网络层（`nn`模块）、优化器（`optim`模块）等丰富的工具和函数，方便搭配各种模型。

### 1.1 查看Torch版本

```python
import torch
print(torch.__version__)
```

用于检测PyTorch安装情况以及是否成功激活CUDA加速。

在Macbook Air M3上正确安装后输出：

```bash
2.7.0
```

在深度学习服务器上正确安装后输出：

```bash
2.6.0+cu124
```

`cu124`这里指CUDA 12.4版本，得到输出即为安装成功。

### 1.2 张量（`tensor`）

张量可以理解为`numpy.ndarray`，但更方便在GPU上运算。

#### 1.2.1 初始化

`torch.empty`用于生成空张量，`dtype=`用于指定数据类型，也可以不进行指定，类型会等于`torch.float32`。

```python
x = torch.empty(3, 3, dtype=torch.long) # Empty 3x3 Tensor
print(x)
```

```python
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
```

`torch.ones`用于生成用1初始化的张量。

```python
x = torch.ones(3, 3) # 3x3 Tensor of ones
print(x)
```

```python
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])
```

`torch.rand`用于生成随机数填充的张量。

```python
x = torch.rand(3, 3, dtype=torch.double) # Random 3x3 Tensor
print(x)
```

```python
tensor([[0.2974, 0.9091, 0.4779],
        [0.8316, 0.9610, 0.2925],
        [0.7015, 0.2892, 0.5892]], dtype=torch.float64)
```

#### 1.2.2 运算与转换

张量之间可以进行四则运算。

```
x = torch.rand(3, 3)
y = torch.rand(3, 3)

z = x + y
```

因为`torch.tensor`是`numpy.ndarray`的拓展，两者可以相互转换。

```python
np_array = torch_tensor.numpy()
torch_tensor = torch.from_numpy(np_array)
```

#### 1.2.3 Autograd

将`require_grad`设置为`True`，利用Autograd模块，可以自动对所有的步骤进行求导，方便地进行反向传播。

```Python
x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
z = y * y * 3
out = z.mean()

out.backward()
print(x.grad)
```

首先，代码将张量初始化为$x =\begin{bmatrix}1 , 1 \\1 , 1\end{bmatrix}$，其次执行$y_{ij} = x_{ij} + 2$与$z_{ij} = (y_{ij})^2 \cdot 3$，张量变为$z = \begin{bmatrix} 27 ， 27 \\ 27 ， 27 \end{bmatrix}$。

接着，通过代码`out = z.mean()`，执行求平均值，$\text{out} = \frac{1}{4} \sum_{i=0}^{1} \sum_{j=0}^{1} z_{ij}=27$。

最后，代码`out.backward()`执行反向传播梯度计算，计算$\frac{\partial \text{out}}{\partial x_{ij}}$ ，`x.grad`中存储的正是$\frac{\partial \text{out}}{\partial x_{ij}}$的值，即为$\text{out}$对$x$的梯度。

> 参考文章：[CSDN：深度学习框架：PyTorch使用教程 ！！](https://blog.csdn.net/leonardotu/article/details/147402400)

## 2 Python的类与继承

### 2.1 无继承的类

```python
class Animal:
    def __init__(self):
        print("Init")
    def shout(self):
        print('Animal shouts')
a = Animal()
a.shout()
```

事实上，Python只是规定，无论是构造方法还是实例方法，都必须至少包含一个参数，将其命名为`self`，主要是因为约定俗成。实际运行中，`self`代表着方法的调用者。

名字为`__init__`的函数构造方法，将在类初始化时运行。

### 2.2 带继承的类

```python
class Animal:
    def __init__(self, name):
        self._name = name

    def shout(self): # 一个通用的叫方法
        print('{} shouts'.format(self._name))

    @property
    def name(self):
        return self._name

a = Animal('monster')
a.shout()

class Cat(Animal):
    pass

cat = Cat('garfield')
cat.shout()
print(cat.name)
```

> `pass`用于占位，不执行任何操作。

可以看到`Cat`继承了`Animal`类的方法。

这里可以看到`name`函数使用了`@property`装饰器。

另外，在子类在继承父类之后，定义同名的函数即可进行覆写。

> 参考文章：[博客园: Python-类的继承](https://www.cnblogs.com/ygbh/p/17556532.html)

### 2.3 `@property`装饰器

```python
class C(object):
    @property
    def x(self):
        "I am the 'x' property."
        return self._x
     
    @x.setter
    def x(self, value):
        self._x = value
         
    @x.deleter
    def x(self):
        del self._x
```

`@property`修饰器用于修饰类，`@x.setter`用于声明某个参数的设置属性值的函数，`@x.getter`用于声明某个参数的获取属性值的函数，`@x.deleter`用于删除某属性值。

计划下周Python部分补一下Python装饰器相关语法。

> 参考文章：[知乎: Python装饰器中@property使用详解](https://zhuanlan.zhihu.com/p/16053072064)

### 2.4 超类

```python
class A:
    def __init__(self):
        self.n = 2

    def add(self, m):
        print('self is {0} @A.add'.format(self))
        self.n += m


class B(A):
    def __init__(self):
        self.n = 3

    def add(self, m):
        print('self is {0} @B.add'.format(self))
        super().add(m)
        self.n += 3

if __name__=="__main__":
		b = B()
		b.add(2)
		print(b.n)
```

```python
self is <__main__.B object at 0x105b8d010> @B.add
self is <__main__.B object at 0x105b8d010> @A.add
8
```

`super()`用于执行父类的函数，在Python3中`super().add()`等价于Python2中的`super(B, self).add()`。

> 需要注意的是，当我们调用 `super()`的时候，实际上是实例化了一个 `super`类。
>
> `super`实际上做的事简单来说就是：提供一个 **MRO** 以及一个 **MRO** 中的类 **C** ， **super()** 将返回一个从 **MRO** 中 **C** 之后的类中查找方法的对象。也就是说，查找方式时不是像常规方法一样从所有的 **MRO** 类中查找，而是从 **MRO** 的 tail 中查找

> 参考文章：[菜鸟教程: Python super 详解](https://www.runoob.com/w3cnote/python-super-detail-intro.html)

## 3 神经网络

### 3.1 构建神经网络模块

#### 3.1.1 `nn.Sequential()`定义

**优点**：简单、易读，顺序已经定义好，不需要再写forward。

**缺点**：丧失灵活性

##### 3.1.1.1 子模块定义方式

```python
import torch.nn as nn
net = nn.Sequential(
    nn.Linear(784, 10)
)
print(net)
```

```python
Sequential(
  (0): Linear(in_features=784, out_features=10, bias=True)
)
```

##### 3.1.1.2 子模块有序字典定义方式

```python
import collections
import torch.nn as nn
net = nn.Sequential(collections.OrderedDict([
        ('fc1', nn.Linear(784, 10))
    ]))
print(net)
```

```python
Sequential(
  (fc1): Linear(in_features=784, out_features=10, bias=True)
)
```

#### 3.1.2 `nn.ModuleList()`定义

```python
import torch.nn
net = nn.ModuleList([nn.Linear(784, 10)])
# net.append(nn.Linear(256, 10)) # 可以添加新的层
# print(net[-1]) # 可以通过索引访问
print(net)
```

```python
ModuleList(
  (0): Linear(in_features=784, out_features=10, bias=True)
)
```

定义完成之后只是`nn.ModuleList`，并不是模型，需要在模型中指定`forward()`函数。

```python
class model(nn.Module):
  def __init__(self, ...):
    super().__init__()
    self.modulelist = ...
    ...
    
  def forward(self, x):
    for layer in self.modulelist:
      x = layer(x)
    return x
```

#### 3.1.3 `nn.ModuleDict()`定义

```python
import torch.nn as nn
net = nn.ModuleDict({
    'fc1' : nn.Linear(784, 10)
})
# print(net['fc1']) # 访问特定层，字典索引
# print(net.fc1) # 访问特定层，成员索引
print(net)
```

```
Linear(in_features=784, out_features=10, bias=True)
ModuleDict(
  (fc1): Linear(in_features=784, out_features=10, bias=True)
)
```

同`nn.ModuleList`，定义完成之后只是`nn.ModuleDict`，并不是模型，需要在模型中指定`forward()`函数。

#### 3.1.4 继承`nn.Module`以定义模块

定义一个模块的基本模板如下：

```python
import torch.nn as nn
import torch.nn.functional as F
 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10) # 定义一个全连接层：输入维度 784，输出维度 10
 
    def forward(self, x):
        x = x.view(-1, 784) # 将输入 x 展平成 (batch_size, 784)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
 
model = Model()
print(model)
```

> 参考文章：[CSDN: 【PyTorch】4-模型定义（Sequential、ModuleList/ModuleDict、模型块组装、修改模型、模型保存和读取）](https://blog.csdn.net/m0_65787507/article/details/138464752)

### 3.2 实现全连接神经网络（FCNN）

通过实现全连接神经网络（Fully-Connected Neural Network），熟悉Pytorch实现神经网络的代码架构以及必须组成部分。

```python
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
```



#### 3.2.1 数据集

为了比较方便地验证神经网络的效果，采用MNIST数据集。

MNIST包含70,000张手写数字图像： 60,000张用于训练集，10,000张用于测试集。**图像都为$28\times28$的灰度图**，并且已经居中处理，以减少预处理操作。后续使用该数据集不再赘述背景，在Pytorch中，该数据集可以自动下载，将`dataset.MNIST(download=True)`时可以在指定目录不存在数据集时将数据集自动下载进第一个参数设置的目录中，`transform=transform`指定了变换为定义的`transform`。

```python
train_set = datasets.MNIST('dataset/mnist/', train=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_set = datasets.MNIST('dataset/mnist/', train=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
```

```python
transform = transforms.Compose([
    transforms.ToTensor(),
])
```

`DataLoader`设置中，第一个参数指定了数据集，`batch_size=64`将批次大小为64；`shuffle=True`指在每个epoch开始时，会将训练图片顺序打乱，有助于泛化；当`train=True`时将数据集设定为训练集，反之则为测试集。

`transforms.ToTensor()`是PyTorch中用于图像数据预处理的关键函数，方便的将`PIL.Image`转换为了`torch.tensor`。

#### 3.2.2 实现FCNN网络结构

FCNN的网络结构由输入层、隐藏层、输出层构成，根据FCNN网络结构的特点，每个神经元都应该与前一层的所有节点相连。

MNIST数据集为$28\times28$的灰度图像，则将图像展平到一维后应该有$28\times28=784$个特征，即输入层应该有784个特征。

而手写数字数据集的结果应该包含0~9，10个类别，即输出应该有10个特征。

```python
class FCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.moduledict = nn.ModuleDict({
            'fc1' : nn.Linear(784, 128), # All img in MINST are 28x28, 28*28=784
            'fc2' : nn.Linear(128, 128),
            'relu': nn.ReLU(),
            'out' : nn.Linear(128, 10)
        })
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.moduledict.fc1(x)
        x = self.moduledict.relu(x)
        x = self.moduledict.fc2(x)
        x = self.moduledict.relu(x)
        return self.moduledict.out(x)
```

需要解释的是`x.view(x.size(0), -1)`，具体来说，`x.view(x.size(0), -1)`的含义是，`x.size(0)`为张量`x`的第一个维度的大小（即，如果`x`是一个形状为`(a, b, c)`的张量，那么`x.size(0)`就等于`a`）。在大多数情况下，这代表了批处理中的样本数或者是序列的长度，取决于上下文。-在`.view()`函数中，-1是一个特殊的值，表示该维度的大小会自动计算，以便保持总元素数不变。换句话说，PyTorch会根据其他维度的大小和总元素数来推断出-1应该代表的具体数值。

在这里，每个batch输入FCNN时，`tensor.shape`为`torch.Size([64, 1, 28, 28])`，经过`x.view(x.size(0), -1)`后特征被展平为`torch.Size([64, 784])`，即每张图像都被展平为一维，第二个维度为batch。

[Reference]: https://blog.csdn.net/a8039974/article/details/142183557

#### 3.2.3 训练过程

```python
from numpy import shape
from tqdm import tqdm


model = FCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
epoch_num = 5

model.train()

for epoch in range(epoch_num):
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epoch_num}'):
        optimizer.zero_grad() # 梯度清0
        output = model(images) # 训练单个Epoch
        loss = criterion(output, labels) # 损失函数正向计算
        loss.backward() # 反向传播
        optimizer.step() # 优化器计算参数
```

```bash
Epoch 1/5: 100%|██████████| 938/938 [00:01<00:00, 580.48it/s]
Epoch 2/5: 100%|██████████| 938/938 [00:01<00:00, 612.49it/s]
Epoch 3/5: 100%|██████████| 938/938 [00:01<00:00, 591.22it/s]
Epoch 4/5: 100%|██████████| 938/938 [00:01<00:00, 533.77it/s]
Epoch 5/5: 100%|██████████| 938/938 [00:01<00:00, 628.34it/s]
```

`nn.CrossEntropyLoss()`使用交叉熵函数作为损失函数，`torch.optim.Adam(model.parameters(), lr=0.001)`将优化器设定为Adam优化器，Epoch设定为5，并使用tqdm库优化训练进度可视化，使其更加可读。

这样就实现了一个最简单的模型训练过程。

> 参考文章：[博客园: pytorch简单识别MNIST的全连接神经网络](https://www.cnblogs.com/grasp/p/18540403)


## 总结

本周对Python的一些不熟悉的概念，比如Python下的类与继承的实现、类变量作用域（未提及）、Pytorch概念基础、模型的几种定义方式进行了熟悉，尝试利用MNIST手写数据集和最简单的FCNN训练了一个最简单的模型。

这周发现有遗留不熟悉的知识点：Mixin、修饰器和神经网络的优化器、图像处理的transform等。Mixin和修饰器不是非常紧迫，暂时搁置；优化器和transform的内容在下周的学习中补齐。

下周暂定计划是对本周训练的模型进行验证，尝试类别输出与Softmax概率输出等，计算各类评估参数。尝试修改网络结构，实现CNN。找到可以用于简单时序模型的数据，开启各类时序模型的学习。
