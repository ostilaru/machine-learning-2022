from audioop import mul
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
class LogisticRegression:

    def __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
        """The logistic sigmoid function"""
        

    # sigmoid一阶导数
    def dsigmoid(self,x):
        return self.sigmoid(x) * (1-self.sigmoid(x))

    def loss(self, y_r, y_p):
        return y_r*np.log(y_p) + (1 - y_r)*np.log(1 - y_p )

    # 负对数似然函数 loss_f，np.nan_to_num()函数是为了防止出现log0的情况，
    # 最后的loss_f还除以了一个训练数据总数即y_r.shape[0]
    def loss_f(self, y_r, y_p):
        return -np.sum(np.nan_to_num(self.loss(y_r, y_p))) / y_r.shape[0]

    def fit(self, X, y, lr=0.01, tol=1e-7, max_iter=1e7):
        """
        Fit the regression coefficients via gradient descent or other methods 
        """
        # 首先对X进行标准化
        X = np.array(X)
        n,m = X.shape
        y = np.array(y)
        mu = np.mean(X, 0)     # 列的均值
        sigma = np.std(X, 0)   # 列的标准差
        X = (X - mu)/sigma
        # add ones
        X = np.c_[X, np.ones((n,1))]
        # int w
        w = np.random.randn(m+1)    # 标准正态的随机浮点矩阵
        loss = []
        iters = []
        # gradient descent
        for i in range(max_iter):
            y_p = self.sigmoid(np.dot(w, np.transpose(X)))
            if i%1000 == 0:
                print(f'itr=={i}, loss=={self.loss_f(y, y_p)}')
            loss.append(self.loss_f(y, y_p))
            iters.append(i)
            dw = np.dot((y_p - y), X)/n # 求dw
            w = w - lr*dw               # 梯度下降 lr为学习率
            if w <= tol:                # 梯度足够小时，停止迭代
                break
        y_p = self.sigmoid(np.dot(w, np.transpose(X)))
        print(f'finally loss:{self.loss_f(y, y_p)}')
        plt.plot(iters, loss)
        return w, mu, sigma

    # sign函数是一个阶跃函数，将小于0.5的数映射到0，大于0.5的数映射到1
    def sign(self, y):
        for i in range(y.shape[0]):
            if y[i]<0.5:
                y[i] = 0
            else:
                y[i] = 1
        return y
        

    """
    首先对X数据进行一些预处理,预处理的方式是进行标准化,
    标准化的方法是对X的每一个属性减去它的均值除以它的方差,
    其中均值与方差是要保留下来传递给后面的测试函数的,
    即用训练集的分布估计整体的数据分布，并在测试集数据中进行同样的标准化处理
    之后如上所说加入一列1向量并初始化w,用梯度下降得到最终参数w。
    注意其中dw = np.dot((y_p-y),X)/n 是除以了一个n的,
    这是因为我们设的loss_f除以了一个n,并不影响梯度下降的结果
    但能有效的避免数值计算上出现的一些问题，使用标准化的预处理也是为了这个目的
    """    

    def get_acc(self, y_r, y_p):
        y_p = self.sign(y_p)
        s = 0
        for i in range(y_r.shape[0]):
            if y_r[i] == y_p[i]:
                s += 1
        return s/y_r.shape[0]

    

    def predict(self, X, y, w, mu, sigma):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        """
        """
        函数非常简单，先对测试集数据进行与训练集相同的预处理，
        进行标准化并加一列1向量,再通过train得到的参数得到对数机率,
        并用sign及int2str函数得到最终要输出的估计结果
        """
        # 得到测试结果并输出
        X = np.array(X)
        n,m = X.shape
        X = (X - mu)/sigma
        # add ones
        X = np.c_[X, np.ones((n,1))]
        y = np.array(y)
        y_p = self.sigmoid(np.dot(w, np.transpose(X)))
        acc = self.get_acc(y,y_p)
        print(f'acc = {acc}')
        return y_p
        

    
    
        