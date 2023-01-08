# 实验二 SVM 支持向量机

- PB20111635
- 蒋磊

## 实验目的

+ 熟悉SVM的原理并实现
+ 了解机器学习及模型学习及评估的基本方法

## 实验原理

### SVM模型

**支持向量机**是一种二分类模型，它的基本模型是定义在特征空间上的间隔最大的线性分类器。

间隔可以使用 $ \gamma = \frac{2}{||w||} $ 来表示，这样求解 `SVM` 模型可以变成下面的优化问题：
$$
\mathop{\max}_{\b{w},b} \frac{2}{\| \b{w} \| } \\
\text{s.t.} \, y_i(\b{w}^T \b{x_i} + b) \geq 1  \quad i=1, ..., m
$$
等价于：
$$
\mathop{\min}_{\b{w},b} \frac{1}{2} \| \b{w} \|^{2} \\
\text{s.t.} \, y_i(\b{w}^T \b{x_i} + b) \geq 1  \quad i=1,\dots,m
$$
上面的模型只能解决线性可分的问题，为了解决有部分数据点线性不可分的情况，需要增加软间隔， 软间隔允许某些样本不满足约束 $y_i(\b{w}^T \b{x_i} + b) \ge 1$. 为了使不满足约束的样本尽可能少, 优化目标可以写为
$$
\min_{\b{w},b}\ \frac{1}{2}\|\b{w}\|^{2}+C\sum_{i=1}^{m}l_{0/1}(y_{i}(\b{w}\T\b{x}_{i}+b)-1)
$$
其中 $C>0$ 是一个常数，$l_{0/1}$ 是 “0/1损失函数”
$$
l_{0/1}=\begin{cases}1, & \text{if } z<0;\\0, & \text{otherwise}.\end{cases}
$$
由于 $l_{0/1}$ 非凸、非连续，数学性质不太好，下面使用如下的 hinge loss 函数来替代它：
$$
l_{\text{hinge}}(z)=\max(0, 1-z)
$$
如此， 优化问题变成：
$$
\min_{\b{w},b}\ \frac{1}{2}\|\b{w}\|^{2}+C\sum_{i=1}^{m}\max(0, 1-y_{i}(\b{w}\T\b{x}_{i}+b))
$$

### 模型学习方法

### （1）梯度下降法

为了求解模型中的参数 $\b{w}$ 和 $b$，我们可以使用**梯度下降法**.

记要优化的式子为 $L$, 记 $\xi_{i}=1-y_{i}(\b{w}\T\b{x}_{i}+b)$, 则
$$
\begin{aligned}
\frac{\partial L}{\partial \b{w}}&=\b{w}-C\sum_{\xi_{i}\ge 0}y_{i}\b{x}_{i}\\
\frac{\partial L}{\partial b}&=-C\sum_{\xi_{i}\ge 0}y_{i}
\end{aligned}
$$
**梯度下降法：**
$$
\begin{aligned}
&\text{while}\, \|  \frac{\partial L}{\partial \b{w}} \| +\|\frac{\partial L}{\partial b} \| >\delta\, \text{do}\\
&\quad \text{for }i=1\text{ to } m \text{ do}\\
&\quad\quad \xi_{i}\leftarrow1-y_{i}(\b{w}\T_{t}\b{x}_{i}+b_{t})\\
&\quad \b{w}_{t+1}\leftarrow \b{w}_{t} - \eta(\b{w}_{t}- C \sum_{\xi_{i}\ge 0}y_{i}\b{x}_{i})  \\
&\quad b_{t+1} \leftarrow b_{t}  -  \eta (-C \sum_{\xi_{i}\ge 0}y_{i}) \\
&\text{end while}
\end{aligned}
$$

#### 梯度下降法的实验结果

在梯度下降法中，我生成的数据是 20 维，100000 组数据，其中 99000 组数据作为训练集，剩下 10000 组作为测试集，学习率为 0.001，最大迭代次数为 500（经过测试模型大概在跑完 300 个 epoch 后收敛），惩罚系数为 1，初始生成数据的错标率我没有改动，也就是助教设置的 4%左右。

~~~python
X_data, y_data, mislabel = generate_data(20, 100000) 

# split data
X_train = X_data[0:90000]
y_train = y_data[0:90000]
X_test = X_data[90001:]
y_test = y_data[90001:]

# constrcut model and train (remember record time)
model1 = SVM1(20, learning_rate=0.001, max_iter=500, C=1)
model1.fit(X_train, y_train, val_data=(X_test, y_test))
~~~

<img src="/Users/jianglei/somebook_and_course/ML/LAB/LAB1_PB20111635_蒋磊/截屏2022-10-28 17.57.54.png" alt="截屏2022-10-28 17.57.54" style="zoom: 150%;" />

如图所示，最终在此参数条件下，模型的训练集与测试集均可以达到大约 90% 的正确率。

但由于初始错误率较低，后不断振荡最终收敛，这里有过拟合的的风险，于是我将参数进行了一些修改，由于是采取了软间隔的模型，最初的惩罚系数 C=1，我认为设置的有些过大了，这将导致间隔较小，所以我将惩罚系数 C 改为 0.001，故意将间隔放大，以容忍更多的错误，不过这样调参的合理性仍有待考证，最终能够得到这样的结果：

<img src="/Users/jianglei/somebook_and_course/ML/LAB/LAB1_PB20111635_蒋磊/截屏2022-10-28 18.02.44.png" alt="截屏2022-10-28 18.02.44" style="zoom:67%;" />

可以看到，训练集的错误率在 3.8%左右，测试集的错误率在 4.1%左右，那么正确率大约在 96%，这与生成数据时设置的错标率是非常接近的。

为了更直观的展示决策边界，我将生成的数据集改为了二维，以便利用散点图进行可视化：

 <center class="half">
     <img src = "/Users/jianglei/somebook_and_course/ML/LAB/LAB1_PB20111635_蒋磊/截屏2022-10-28 17.12.16.png"align=left style="zoom: 40%;" />
     <img src = "/Users/jianglei/somebook_and_course/ML/LAB/LAB1_PB20111635_蒋磊/截屏2022-10-28 17.11.47.png"align=right style="zoom: 40%;" />





























可见软间隔模型在二维的情况下的分类情况是比较理想的，正确率能够达到 95%左右，这与之后调用 sklearn 库的结果非常接近。

### （2）SMO 序列最小优化算法

SMO 算法的原理我认为和坐标轮换法非常类似，只不过它每次选择两个坐标，而且SVM 的对偶函数具有两个约束条件，而坐标轮换法适用于求无约束条件的情况。

我们知道 SVM 的对偶函数如下：

![截屏2022-10-28 18.31.42](/Users/jianglei/somebook_and_course/ML/LAB/LAB1_PB20111635_蒋磊/截屏2022-10-28 18.31.42.png)

求解步骤如下：

![截屏2022-10-28 18.32.42](/Users/jianglei/somebook_and_course/ML/LAB/LAB1_PB20111635_蒋磊/截屏2022-10-28 18.32.42.png)

我在实现 SMO 算法时，由于觉得实现起来稍微有些麻烦，所以没有采取启发式寻找向量$a_i 和 a_j$，而是用了随机选取，这样做会导致效率低不少，于是只能通过暴力增加迭代次数来换取正确率的提升。

这里我所选取的参数是最大迭代次数 `maxiter=1000`，惩罚系数`C=1`，由于随机选取导致的效率低下，我实现的 SMO 需要较长时间才能收敛，于是我修改了数据集大小，这次我只生成了 20 维，10000 组数据。由于在实现时出现了不少 bug 所以我保留了不少注释，希望助教在检查时多多包涵。

```python
# generate data
X_data, y_data, mislabel = generate_data(20, 10000) 
# print(X_data)
# print(y_data)

# split data
X_train = X_data[0:9000]
y_train = y_data[0:9000]
X_test = X_data[9001:]
y_test = y_data[9001:]

# constrcut model and train (remember record time)
model2 = SVM2(20, maxiter=1000, C=1)
model2.fit(X_train, y_train, val_data=(X_test, y_test))
```

#### SMO 的实验结果

<img src="/Users/jianglei/somebook_and_course/ML/LAB/LAB1_PB20111635_蒋磊/截屏2022-10-28 17.52.45.png" alt="截屏2022-10-28 17.52.45" style="zoom:150%;" />

可以看到，在最初的 200 个 epoch 中，SMO 效率较高，之后出现了振荡直至收敛，最终正确率在 90% 左右，这略低于梯度下降法的正确率，不过这个结果我是可以接受的。

### （3）调用 sklearn 库进行对比

由于对 sklearn 库的使用并不是非常熟练，这里我只是简单的得到了调用标准库的正确率：

![截屏2022-10-28 18.47.12](/Users/jianglei/somebook_and_course/ML/LAB/LAB1_PB20111635_蒋磊/截屏2022-10-28 18.47.12.png)

可以看到调用标准库得到的准确率在 95%左右。

## 总结

本次实验对于我来说难度较大，尤其是 SMO 算法的实现，在完成实验时对理论推到进行了仔细地查阅和推导，但仍遇到了不少问题，花费了不少时间在本次实验上，不过确实是对 SVM 支持向量机的原理有了比较清楚的认识，总的来说收获很大。不过还是建议助教可以在实验文档中多增加一些提示和教学，帮助同学们更加轻松地完成实验。