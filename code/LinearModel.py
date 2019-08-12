'''

Author : 苏冠旭
update date : 20190717

本模块目前包含：
    1. 由于sklearn自带的Logistics回归模型在l1惩罚项下拟合较慢，本模块针对几个包含l1罚项的线性回归模型编写了tensorflow的实现，增加了拟合
    其调用方式基本与sklearn中的模型完全相同，即支持fit()、predict()、predict_log_proba()、predict_proba()方法，包含coef_、intercept_等属性。这部分包括：
        1) Lasso二分类回归模型LassoClassifier、LassoClassifierCV
        2) ElasticNet二分类回归模型EnetClassifier、EnetClassifierCV
        3) Group Lasso二分类回归模型GroupLassoClassifier、GroupLassoClassifierCV
            注:  
                a) GroupLasso要求变量按照分组进行排序，同一组的变量排在一起
                b) 组别参数K的说明 : a dictionary K of number of features in each group
        4) Alpha Lasso二分类回归模型AlphaLassoClassifier、AlphaLassoClassifierCV
            注:
                a)Alpha Lasso将参数beta_index指定的那些系数不加入l1罚项，保证beta_index指定的那些系数一定可以入模
                b)beta_index应当是range(X_train.shape[1])的子集
        5) Positive Lasso二分类回归模型PosLassoClassifier、PosLassoClassifierCV
            注：
                a)Positive Lasso限制所有的系数都为非负值。适用于WOE场景。
                b)目前原理是每一次迭代之前将负值系数置0

    2. 针对需要划分行业的lasso建模问题，设计了将行业indicator加入所有的回归系数中，即加入行业与所有变量的交互效应，再用lasso自动筛选
    筛选过后将得到的回归系数还原到各个行业下。这样的好处是建立统一的回归模型，不用分行业建立单独的模型，节约了自由度，利用了更多的数据信息。
    这部分的函数主要包括：
        1) MultilevelVariable：
            将indicator_col二值化，并将每一个level和X_cols中的每一列做交互项。同时增加二值化对应的常数项
            Input:
                df : 数据框
                indicator_col : 需要进行二值化的列，字符串
                X_cols : 自变量对应的列名，list
            Output:
                [0] : 处理之后的数据框
                [1] : 更新后的自变量对应的列名，list
        2) MultilevelCoef ： 
            和MultilevelVariable配套使用，将Multilevel Model得到的系数还原到各个行业中
            Input:
                coefs : 模型得到的系数，不包括常数项
                intercept : 模型得到的常数项
                new_X_cols : MultilevelVariable返回的feature对应的列名，list
                level_names : 分维变量的各个维度的名称组成的list，比如list(set(df[indicator_col]))
                indicator_col : 原数据框中分维度的列对应的列名，与MultilevelVariable中的indicator_col相同，字符串类型
            Output:
                一个数据框，列为各个行业，行为各个系数和常数项
        3) byIndustryModel : 
            将indicator_col二值化，并将每一个level和X_cols中的每一列做交互项。同时增加二值化对应的常数项。
            以此来建立Lasso模型，进行变量筛选。返回建模结果。
            Input : 
                df : 数据框
                X_cols : 自变量对应的列名，list
                y_col : 因变量对应的列名，字符串
                industry_col : 需要进行二值化的列，字符串
                test_size : 测试集比例，float
                model : 一个类sklearn模型，包含model.fit()和model.predict()方法，包含model.coef_和model.intercept_属性
            Output : 
                [0] : 一个数据框，列为各个行业，行为各个系数和常数项，由Lasso返回结果加工得到
                [1] : AUC
                [2] : 拟合之后的model
    3. 在回归模型建立之后，还需要根据KS曲线选取最优的阈值。本模块提供KS曲线的绘制函数PlotKS(preds, labels, n=100, asc=0)


-------------------------------------------------------------------------------------------
-------------------------------------2019-05-23升级说明-------------------------------------
-------------------------------------------------------------------------------------------

增加功能：
    1. 增加部分约束的Alpha Lasso，允许只对部分系数进行l1惩罚，不加惩罚的系数index用beta_index参数指明
    2. 所有模型增加批训练功能，应对数据量太大而运算资源不够的问题。
        1）批训练的batch_size默认值为1024，如果将batch_size改为X_train.shape[0]，则等价于全量训练
        2）由于批训练每次迭代只训练一部分数据，最大迭代次数max_iter应该相应增大
        3）为了确保模型收敛，请务必在模型训练结束后使用模型的plotLoss()方法查看损失函数变动图
    3. 所有模型class增加说明文档，在jupyter中，可以通过将光标置于model class括号中，按Shift+Tab查看


-------------------------------------------------------------------------------------------
-------------------------------------2019-06-06升级说明-------------------------------------
-------------------------------------------------------------------------------------------

增加功能：
    1. 正系数Lasso回归Positive Lasso。在每一次迭代之前将所有的负系数置0。
       class名为PosLassoClassifier和PosLassoClassifierCV


-------------------------------------------------------------------------------------------
-------------------------------------2019-07-10升级说明-------------------------------------
-------------------------------------------------------------------------------------------

增加功能：
    1. 对于LassoClassifier,EnetClassifier,PosLassoClassifier,
           LassoClassifierCV,EnetClassifierCV,PosLassoClassifierCV 等六个Class，   
       增加Y_hat_track属性，记录模型收敛后100次迭代中训练集上各个样本的Y预测概率值
       用于评价收敛之后模型在随机梯度下降中的稳定性。
    2. 优化LassoClassifierCV,EnetClassifierCV,PosLassoClassifierCV的跟踪显示
    3. 更改所有模型的迭代步长缩减率，使之更加合理。


-------------------------------------------------------------------------------------------
-------------------------------------2019-07-12升级说明-------------------------------------
-------------------------------------------------------------------------------------------

增加功能：
    1. 对于LassoClassifier,EnetClassifier,PosLassoClassifier,
           LassoClassifierCV,EnetClassifierCV,PosLassoClassifierCV 等六个Class，   
       增加beta_track属性，记录模型收敛后100次迭代中训练集上各个样本的系数值
       用于评价收敛之后模型在随机梯度下降中的稳定性。
       增加beta_mean参数，当beta_mean=True时，系数的估计采取收敛后100次迭代的均值
       
       
-------------------------------------------------------------------------------------------
-------------------------------------2019-07-17升级说明-------------------------------------
-------------------------------------------------------------------------------------------

增加功能：
    1. 修复predict_proba中的部分bug

-------------------------------------------------------------------------------------------
-------------------------------------2019-08-06升级说明-------------------------------------
-------------------------------------------------------------------------------------------

增加功能：
    1. 对于LassoClassifier,EnetClassifier,PosLassoClassifier,
           LassoClassifierCV,EnetClassifierCV,PosLassoClassifierCV 等六个Class，
       增加start_point参数。如果设置为'ridge',则会选取线性岭回归估计量作为系数迭代的初始值
       否则，选取随机正态分布作为系数迭代初始值。根据https://www.tuicool.com/articles/mAbiq2，
       用最小二乘法估计量作为初始值可以有效减少不收敛的发生，加快收敛速度。


-------------------------------------------------------------------------------------------
-------------------------------------2019-08-08升级说明-------------------------------------
-------------------------------------------------------------------------------------------

增加功能：
    1. 对于LassoClassifier,EnetClassifier,PosLassoClassifier,
           LassoClassifierCV,EnetClassifierCV,PosLassoClassifierCV 等六个Class，
       增加Q参数。如果设置为一个小于1的非负值,则损失函数计算均值时会除去大于Q quantile的元素
       减少异常值的影响，使得模型更加稳健。

'''




import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import copy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import datetime



#############Model Classes###############
class Models(object):
    def __init__(self,max_iter=100,batch_size=1024,standardize=False):
        self.max_iter=max_iter
        self.batch_size=batch_size
        self.standardize = standardize

    def predict_log_proba(self,dataX):
        linearPredictor = np.matmul(dataX,self.coef_)+self.intercept_
        return(linearPredictor)

    def predict_proba(self,dataX):
        linearPredictor = np.matmul(dataX,self.coef_)+self.intercept_
        a = pd.DataFrame(linearPredictor)
        a.loc[a.index[a.loc[:,0]<-40],0] = -40
        a.loc[a.index[a.loc[:,0]>40],0] = 40
        linearPredictor = a[0].values
        prob = 1/(1+np.exp(-linearPredictor))
        return(prob)

    def predict(self,dataX):
        linearPredictor = np.matmul(dataX,self.coef_)+self.intercept_
        a = pd.DataFrame(linearPredictor)
        a.loc[a.index[a.loc[:,0]<-40],0] = -40
        a.loc[a.index[a.loc[:,0]>40],0] = 40
        linearPredictor = a[0].values
        prob = 1/(1+np.exp(-linearPredictor))
        labels = np.array(prob>0.5, dtype = np.int)
        return(labels)

    def plotLoss(self):
        fig,ax=plt.subplots(1,1)
        plt.plot(self.Losses)
        plt.ylabel("Loss")
        plt.rcParams['figure.figsize'] = (6, 3)
        ax.set_title('Losses Track',size=15)
        plt.show()

class LassoClassifier(Models):
    '''
Logistic Regression classifier with l1 penalty.

Parameters
----------
standardize : bool, default: False
    Specifies if each column should be normalized

max_iter : int, default: 100
    Maximum number of iterations of the optimization algorithm.

batch_size : int, default: 1024
    Number of rows of each batch of the batch training.
    If equal to number of rows of training data, then the algorithm
    would be equivalent to full scale training.

lambda_ : float, default: 0.1
    weight of l1 penalty in loss function

beta_mean : boolean, default: False
    If set to True, the coef_ attribute is calculated as the mean
    of coefficients of 100 iterations after converge.

start_point : str, default: 'ridge'
    If set to 'ridge', the start point of coefficient would be ridge
    estimator. Otherwise, the start point would be random
    normal with mean 0.2

Q0 : float, default: 0.98
    If set to be a non-negative value less than 1, the loss function for
    the smaples of lable equals to 0 would be modified to exclude the
    elements greater than quantile Q0 before calculate the mean.

Q1 : float, default: 1.0
    If set to be a non-negative value less than 1, the loss function for
    the smaples of lable equals to 1 would be modified to exclude the
    elements greater than quantile Q1 before calculate the mean.

Attributes
----------
coef_ : array, shape (1, n_features) or (n_classes, n_features)
    Coefficient of the features in the decision function.

    `coef_` is of shape (1, n_features) when the given problem
    is binary.

intercept_ : array, shape (1,)
    Intercept (a.k.a. bias) added to the decision function.

Losses : array
    Array of Loss recorded during training iterations.

Y_hat_track : array, shape (n,100)
    Array of predicted probabilities of training data, tracked
    for 100 iterations after converge.

beta_track : array, shape (n_features,100)
    Array of fitted coeficients tracked for 100 iterations
    after converge.

Methods
-------
fit : optimize the parameters

predict_log_proba : calculate the linear predictor

predict_proba : calculate the probability of 1

predict : calculate the class

plotLoss : plot the trace plot of loss

Examples
--------
>>> from sklearn.datasets import load_iris
>>> from LinearModel import LassoClassifier
>>> X, y = load_iris(return_X_y=True)
>>> model = LassoClassifier().fit(X, y)
>>> model.predict(X[:2, :])
array([0, 0])
>>> model.predict_proba(X[:2, :]).shape
(2, 3)
    '''
    def __init__(self,lambda_=0.1,max_iter=100,batch_size=1024,standardize=False,\
                 beta_mean=False,start_point='ridge',Q1=1,Q0=0.98):
        Models.__init__(self,max_iter,batch_size,standardize)
        self.lambda_=lambda_
        self.beta_mean=beta_mean
        self.start_point = start_point
        self.Q0 = Q0
        self.Q1 = Q1

    def fit(self,dataX,dataY):
        coefs, Losses, Y_HAT, beta_track = mini_lasso(dataX,dataY,self.lambda_,self.max_iter,self.batch_size,\
                                                      self.standardize,y_track=True,start_point=self.start_point,\
                                                      Q0=self.Q0,Q1=self.Q1)
        if self.beta_mean:
            a = pd.DataFrame(np.reshape(np.mean(beta_track,axis=1),dataX.shape[1]))
            zero_index = [_ for _ in a.index if np.abs(a.loc[_,0])<0.0001]
            a.loc[zero_index,0] = 0
            coefs[:-1] = a[0]
        self.coef_ = coefs[:-1]
        self.intercept_ = coefs[-1]
        self.Losses = Losses
        Y_hat_track = np.array(Y_HAT)
        Y_hat_track = np.reshape(Y_hat_track,Y_hat_track.shape[:2])
        self.Y_hat_track = Y_hat_track.T
        self.beta_track = beta_track

class LassoClassifierCV(Models):
    '''
Logistic Regression classifier with l1 penalty with CV.

Parameters
----------
standardize : bool, default: False
    Specifies if each column should be normalized

max_iter : int, default: 100
    Maximum number of iterations of the optimization algorithm.

batch_size : int, default: 1024
    Number of rows of each batch of the batch training.
    If equal to number of rows of training data, then the algorithm
    would be equivalent to full scale training.

cv_method : str, default: 'auc'
    criterion of CV, options include 'auc' and 'f1_score'

beta_mean : boolean, default: False
    If set to True, the coef_ attribute is calculated as the mean
    of coefficients of 100 iterations after converge.

start_point : str, default: 'ridge'
    If set to 'ridge', the start point of coefficient would be ridge
    estimator. Otherwise, the start point would be random
    normal with mean 0.2

Q0 : float, default: 0.98
    If set to be a non-negative value less than 1, the loss function for
    the smaples of lable equals to 0 would be modified to exclude the
    elements greater than quantile Q0 before calculate the mean.

Q1 : float, default: 1.0
    If set to be a non-negative value less than 1, the loss function for
    the smaples of lable equals to 1 would be modified to exclude the
    elements greater than quantile Q1 before calculate the mean.

Attributes
----------
coef_ : array, shape (1, n_features) or (n_classes, n_features)
    Coefficient of the features in the decision function.

    `coef_` is of shape (1, n_features) when the given problem
    is binary.

intercept_ : array, shape (1,)
    Intercept (a.k.a. bias) added to the decision function.

Losses : array
    Array of Loss recorded during training iterations.

lambda_ : float
    selected weight of l1 loss by CV

num_valid_beta : int
    number of non-zero coefficients

Y_hat_track : array, shape (n,100)
    Array of predicted probabilities of training data, tracked
    for 100 iterations after converge.

beta_track : array, shape (n_features,100)
    Array of fitted coeficients tracked for 100 iterations
    after converge.

Methods
-------
fit : optimize the parameters

predict_log_proba : calculate the linear predictor

predict_proba : calculate the probability of 1

predict : calculate the class

plotLoss : plot the trace plot of loss

Examples
--------
>>> from sklearn.datasets import load_iris
>>> from LinearModel import LassoClassifierCV
>>> X, y = load_iris(return_X_y=True)
>>> model = LassoClassifierCV().fit(X, y)
>>> model.predict(X[:2, :])
array([0, 0])
>>> model.predict_proba(X[:2, :]).shape
(2, 3)

    '''
    def __init__(self,max_iter=100,batch_size=1024,cv_method='auc',standardize=False,\
                 beta_mean=False,start_point='ridge',Q1=1,Q0=0.98):
        Models.__init__(self,max_iter,batch_size,standardize)
        self.cv_method=cv_method
        self.beta_mean=beta_mean
        self.start_point=start_point
        self.Q0 = Q0
        self.Q1 = Q1

    def fit(self,dataX,dataY):
        coefs, num_valid_beta, lambda_, Losses, Y_HAT, beta_track = lasso_reg(dataX,dataY,self.max_iter,self.batch_size,\
                                                                              self.cv_method,self.standardize,\
                                                                              start_point=self.start_point,\
                                                                              Q0=self.Q0,Q1=self.Q1)
        if self.beta_mean:
            a = pd.DataFrame(np.reshape(np.mean(beta_track,axis=1),dataX.shape[1]))
            zero_index = [_ for _ in a.index if np.abs(a.loc[_,0])<0.0001]
            a.loc[zero_index,0] = 0
            coefs[:-1] = a[0]
        self.coef_ = coefs[:-1]
        self.intercept_ = coefs[-1]
        self.lambda_ = lambda_
        self.num_valid_beta = num_valid_beta
        self.Losses = Losses
        Y_hat_track = np.array(Y_HAT)
        Y_hat_track = np.reshape(Y_hat_track,Y_hat_track.shape[:2])
        self.Y_hat_track = Y_hat_track.T
        self.beta_track = beta_track

class EnetClassifier(Models):
    '''
ElasticNet Regression classifier with l1 penalty.

Parameters
----------
standardize : bool, default: False
    Specifies if each column should be normalized

max_iter : int, default: 100
    Maximum number of iterations of the optimization algorithm.

batch_size : int, default: 1024
    Number of rows of each batch of the batch training.
    If equal to number of rows of training data, then the algorithm
    would be equivalent to full scale training.

lambda_ : float, default: 0.1
    Weight of regulerization in loss function

l1_ratio_ : ratio of l1 penalty, defaulty: 0.5
    If set to 1, the algorithm is equivalent to lasso
    If set to 0, the algorithm is equivalent to ridge

beta_mean : boolean, default: False
    If set to True, the coef_ attribute is calculated as the mean
    of coefficients of 100 iterations after converge.

start_point : str, default: 'ridge'
    If set to 'ridge', the start point of coefficient would be ridge
    estimator. Otherwise, the start point would be random
    normal with mean 0.2

Q0 : float, default: 0.98
    If set to be a non-negative value less than 1, the loss function for
    the smaples of lable equals to 0 would be modified to exclude the
    elements greater than quantile Q0 before calculate the mean.

Q1 : float, default: 1.0
    If set to be a non-negative value less than 1, the loss function for
    the smaples of lable equals to 1 would be modified to exclude the
    elements greater than quantile Q1 before calculate the mean.

Attributes
----------
coef_ : array, shape (1, n_features) or (n_classes, n_features)
    Coefficient of the features in the decision function.

    `coef_` is of shape (1, n_features) when the given problem
    is binary.

intercept_ : array, shape (1,)
    Intercept (a.k.a. bias) added to the decision function.

Losses : array
    Array of Loss recorded during training iterations.

Y_hat_track : array, shape (n,100)
    Array of predicted probabilities of training data, tracked
    for 100 iterations after converge.

beta_track : array, shape (n_features,100)
    Array of fitted coeficients tracked for 100 iterations
    after converge.

Methods
-------
fit : optimize the parameters

predict_log_proba : calculate the linear predictor

predict_proba : calculate the probability of 1

predict : calculate the class

plotLoss : plot the trace plot of loss

Examples
--------
>>> from sklearn.datasets import load_iris
>>> from LinearModel import EnetClassifier
>>> X, y = load_iris(return_X_y=True)
>>> model = EnetClassifier().fit(X, y)
>>> model.predict(X[:2, :])
array([0, 0])
>>> model.predict_proba(X[:2, :]).shape
(2, 3)
    '''
    def __init__(self,lambda_=0.1,l1_ratio_=0.5,max_iter=100,batch_size=1024,standardize=False,beta_mean=False,\
                 start_point='ridge',Q1=1,Q0=0.98):
        Models.__init__(self,max_iter,batch_size,standardize)
        self.lambda_=lambda_
        self.l1_ratio_=l1_ratio_
        self.beta_mean = beta_mean
        self.start_point = start_point
        self.Q0 = Q0
        self.Q1 = Q1

    def fit(self,dataX,dataY):
        coefs, Losses, Y_HAT, beta_track = mini_enet(dataX,dataY,self.lambda_,self.l1_ratio_,self.max_iter,\
                                                     self.batch_size,self.standardize,y_track=True,\
                                                     start_point=self.start_point,Q0=self.Q0,Q1=self.Q1)
        if self.beta_mean:
            a = pd.DataFrame(np.reshape(np.mean(beta_track,axis=1),dataX.shape[1]))
            zero_index = [_ for _ in a.index if np.abs(a.loc[_,0])<0.0001]
            a.loc[zero_index,0] = 0
            coefs[:-1] = a[0]
        self.coef_ = coefs[:-1]
        self.intercept_ = coefs[-1]
        self.Losses = Losses
        Y_hat_track = np.array(Y_HAT)
        Y_hat_track = np.reshape(Y_hat_track,Y_hat_track.shape[:2])
        self.Y_hat_track = Y_hat_track.T
        self.beta_track = beta_track

class EnetClassifierCV(Models):
    '''
ElasticNet Regression classifier with l1 penalty with CV.

Parameters
----------
standardize : bool, default: False
    Specifies if each column should be normalized

max_iter : int, default: 100
    Maximum number of iterations of the optimization algorithm.

batch_size : int, default: 1024
    Number of rows of each batch of the batch training.
    If equal to number of rows of training data, then the algorithm
    would be equivalent to full scale training.

cv_method : str, default: 'auc'
    criterion of CV, options include 'auc' and 'f1_score'

beta_mean : boolean, default: False
    If set to True, the coef_ attribute is calculated as the mean
    of coefficients of 100 iterations after converge.

start_point : str, default: 'ridge'
    If set to 'ridge', the start point of coefficient would be ridge
    estimator. Otherwise, the start point would be random
    normal with mean 0.2

Q0 : float, default: 0.98
    If set to be a non-negative value less than 1, the loss function for
    the smaples of lable equals to 0 would be modified to exclude the
    elements greater than quantile Q0 before calculate the mean.

Q1 : float, default: 1.0
    If set to be a non-negative value less than 1, the loss function for
    the smaples of lable equals to 1 would be modified to exclude the
    elements greater than quantile Q1 before calculate the mean.

Attributes
----------
coef_ : array, shape (1, n_features) or (n_classes, n_features)
    Coefficient of the features in the decision function.

    `coef_` is of shape (1, n_features) when the given problem
    is binary.

intercept_ : array, shape (1,)
    Intercept (a.k.a. bias) added to the decision function.

Losses : array
    Array of Loss recorded during training iterations.

lambda_ : float
    selected weight of regularization in loss function by CV

l1_ratio_ : float
    selected ratio of l1 norm in regularization by CV

num_valid_beta : int
    number of non-zero coefficients

Y_hat_track : array, shape (n,100)
    Array of predicted probabilities of training data, tracked
    for 100 iterations after converge.

beta_track : array, shape (n_features,100)
    Array of fitted coeficients tracked for 100 iterations
    after converge.

Methods
-------
fit : optimize the parameters

predict_log_proba : calculate the linear predictor

predict_proba : calculate the probability of 1

predict : calculate the class

plotLoss : plot the trace plot of loss

Examples
--------
>>> from sklearn.datasets import load_iris
>>> from LinearModel import EnetClassifierCV
>>> X, y = load_iris(return_X_y=True)
>>> model = EnetClassifierCV().fit(X, y)
>>> model.predict(X[:2, :])
array([0, 0])
>>> model.predict_proba(X[:2, :]).shape
(2, 3)
    '''
    def __init__(self,max_iter=100,batch_size=1024,cv_method='auc',standardize=False,beta_mean=False,\
                 start_point='ridge',Q1=1,Q0=0.98):
        Models.__init__(self,max_iter,batch_size,standardize)
        self.cv_method=cv_method
        self.beta_mean = beta_mean
        self.start_point = start_point
        self.Q0 = Q0
        self.Q1 = Q1

    def fit(self,dataX,dataY):
        coefs, num_valid_beta, lambda_, l1_ratio_, Losses, Y_HAT, beta_track = enet_reg(dataX,dataY,self.max_iter,\
                                                                                        self.batch_size,self.cv_method,\
                                                                                        self.standardize,\
                                                                                        start_point=self.start_point,\
                                                                                        Q0=self.Q0,Q1=self.Q1)
        if self.beta_mean:
            a = pd.DataFrame(np.reshape(np.mean(beta_track,axis=1),dataX.shape[1]))
            zero_index = [_ for _ in a.index if np.abs(a.loc[_,0])<0.0001]
            a.loc[zero_index,0] = 0
            coefs[:-1] = a[0]
        self.coef_ = coefs[:-1]
        self.intercept_ = coefs[-1]
        self.lambda_ = lambda_
        self.l1_ratio_ = l1_ratio_
        self.num_valid_beta = num_valid_beta
        self.Losses = Losses
        Y_hat_track = np.array(Y_HAT)
        Y_hat_track = np.reshape(Y_hat_track,Y_hat_track.shape[:2])
        self.Y_hat_track = Y_hat_track.T
        self.beta_track = beta_track

class PosLassoClassifier(Models):
    '''
Positive Lasso Regression classifier with l1 penalty.
Positive Lasso Regression means that all coefficients
are contrained to be positive.

Parameters
----------
standardize : bool, default: False
    Specifies if each column should be normalized

max_iter : int, default: 100
    Maximum number of iterations of the optimization algorithm.

batch_size : int, default: 1024
    Number of rows of each batch of the batch training.
    If equal to number of rows of training data, then the algorithm
    would be equivalent to full scale training.

lambda_ : float, default: 0.1
    Weight of regulerization in loss function

pos_penalty : float, default: 2
    Weight of negative regulerization in loss function
    If set to 0, the algorithm is equivalent to lasso

beta_mean : boolean, default: False
    If set to True, the coef_ attribute is calculated as the mean
    of coefficients of 100 iterations after converge.

start_point : str, default: 'ridge'
    If set to 'ridge', the start point of coefficient would be ridge
    estimator. Otherwise, the start point would be random
    normal with mean 0.2

Q0 : float, default: 0.98
    If set to be a non-negative value less than 1, the loss function for
    the smaples of lable equals to 0 would be modified to exclude the
    elements greater than quantile Q0 before calculate the mean.

Q1 : float, default: 1.0
    If set to be a non-negative value less than 1, the loss function for
    the smaples of lable equals to 1 would be modified to exclude the
    elements greater than quantile Q1 before calculate the mean.

Attributes
----------
coef_ : array, shape (1, n_features) or (n_classes, n_features)
    Coefficient of the features in the decision function.

    `coef_` is of shape (1, n_features) when the given problem
    is binary.

intercept_ : array, shape (1,)
    Intercept (a.k.a. bias) added to the decision function.

Losses : array
    Array of Loss recorded during training iterations.

Y_hat_track : array, shape (n,100)
    Array of predicted probabilities of training data, tracked
    for 100 iterations after converge.

beta_track : array, shape (n_features,100)
    Array of fitted coeficients tracked for 100 iterations
    after converge.

Methods
-------
fit : optimize the parameters

predict_log_proba : calculate the linear predictor

predict_proba : calculate the probability of 1

predict : calculate the class

plotLoss : plot the trace plot of loss

Examples
--------
>>> from sklearn.datasets import load_iris
>>> from LinearModel import EnetClassifier
>>> X, y = load_iris(return_X_y=True)
>>> model = EnetClassifier().fit(X, y)
>>> model.predict(X[:2, :])
array([0, 0])
>>> model.predict_proba(X[:2, :]).shape
(2, 3)
    '''
    def __init__(self,lambda_=0.1,pos_penalty=2,max_iter=100,batch_size=1024,standardize=False,beta_mean=False,\
                 start_point='ridge',Q1=1,Q0=0.98):
        Models.__init__(self,max_iter,batch_size,standardize)
        self.lambda_=lambda_
        self.pos_penalty=pos_penalty
        self.beta_mean = beta_mean
        self.start_point = start_point
        self.Q0 = Q0
        self.Q1 = Q1

    def fit(self,dataX,dataY):
        coefs, Losses, Y_HAT, beta_track = mini_PosLasso(dataX,dataY,self.lambda_,self.pos_penalty,\
                                                         self.max_iter,self.batch_size,self.standardize,\
                                                         y_track=True,start_point=self.start_point,\
                                                         Q0=self.Q0,Q1=self.Q1)
        if self.beta_mean:
            a = pd.DataFrame(np.reshape(np.mean(beta_track,axis=1),dataX.shape[1]))
            a.loc[a.index[a.loc[:,0].values<=0.0001],0] = 0
            coefs[:-1] = a[0]
        self.coef_ = coefs[:-1]
        self.intercept_ = coefs[-1]
        self.Losses = Losses
        Y_hat_track = np.array(Y_HAT)
        Y_hat_track = np.reshape(Y_hat_track,Y_hat_track.shape[:2])
        self.Y_hat_track = Y_hat_track.T
        self.beta_track = beta_track

class PosLassoClassifierCV(Models):
    '''
Positive Lasso Regression classifier with l1 penalty.
Positive Lasso Regression means that all coefficients
are contrained to be positive.

Parameters
----------
standardize : bool, default: False
    Specifies if each column should be normalized

max_iter : int, default: 100
    Maximum number of iterations of the optimization algorithm.

batch_size : int, default: 1024
    Number of rows of each batch of the batch training.
    If equal to number of rows of training data, then the algorithm
    would be equivalent to full scale training.

cv_method : str, default: 'auc'
    criterion of CV, options include 'auc' and 'f1_score'

beta_mean : boolean, default: False
    If set to True, the coef_ attribute is calculated as the mean
    of coefficients of 100 iterations after converge.

start_point : str, default: 'ridge'
    If set to 'ridge', the start point of coefficient would be ridge
    estimator. Otherwise, the start point would be random
    normal with mean 0.2

Q0 : float, default: 0.98
    If set to be a non-negative value less than 1, the loss function for
    the smaples of lable equals to 0 would be modified to exclude the
    elements greater than quantile Q0 before calculate the mean.

Q1 : float, default: 1.0
    If set to be a non-negative value less than 1, the loss function for
    the smaples of lable equals to 1 would be modified to exclude the
    elements greater than quantile Q1 before calculate the mean.

Attributes
----------
coef_ : array, shape (1, n_features) or (n_classes, n_features)
    Coefficient of the features in the decision function.

    `coef_` is of shape (1, n_features) when the given problem
    is binary.

intercept_ : array, shape (1,)
    Intercept (a.k.a. bias) added to the decision function.

Losses : array
    Array of Loss recorded during training iterations.

lambda_ : float
    selected weight of regularization in loss function by CV

pos_penalty : float
    selected weight of negative regularization in loss function by CV

num_valid_beta : int
    number of non-zero coefficients

Y_hat_track : array, shape (n,100)
    Array of predicted probabilities of training data, tracked
    for 100 iterations after converge.

beta_track : array, shape (n_features,100)
    Array of fitted coeficients tracked for 100 iterations
    after converge.

Methods
-------
fit : optimize the parameters

predict_log_proba : calculate the linear predictor

predict_proba : calculate the probability of 1

predict : calculate the class

plotLoss : plot the trace plot of loss

Examples
--------
>>> from sklearn.datasets import load_iris
>>> from LinearModel import EnetClassifierCV
>>> X, y = load_iris(return_X_y=True)
>>> model = EnetClassifierCV().fit(X, y)
>>> model.predict(X[:2, :])
array([0, 0])
>>> model.predict_proba(X[:2, :]).shape
(2, 3)
    '''
    def __init__(self,max_iter=100,batch_size=1024,cv_method='auc',standardize=False,beta_mean=False,\
                 start_point='ridge',Q1=1,Q0=0.98):
        Models.__init__(self,max_iter,batch_size,standardize)
        self.cv_method=cv_method
        self.beta_mean=beta_mean
        self.start_point = start_point
        self.Q0 = Q0
        self.Q1 = Q1

    def fit(self,dataX,dataY):
        coefs, num_valid_beta, lambda_, pos_penalty, Losses, Y_HAT, beta_track = \
            PosLasso_reg(dataX,dataY,self.max_iter,self.batch_size,self.cv_method,\
                         self.standardize,start_point=self.start_point,Q0=self.Q0,Q1=self.Q1)
        if self.beta_mean:
            a = pd.DataFrame(np.reshape(np.mean(beta_track,axis=1),dataX.shape[1]))
            a.loc[a.index[a.loc[:,0].values<=0.0001],0] = 0
            coefs[:-1] = a[0]
        self.coef_ = coefs[:-1]
        self.intercept_ = coefs[-1]
        self.lambda_ = lambda_
        self.pos_penalty = pos_penalty
        self.num_valid_beta = num_valid_beta
        self.Losses = Losses
        Y_hat_track = np.array(Y_HAT)
        Y_hat_track = np.reshape(Y_hat_track,Y_hat_track.shape[:2])
        self.Y_hat_track = Y_hat_track.T
        self.beta_track = beta_track


class GroupLassoClassifier(Models):
    '''
GroupLasso Regression classifier with l1 penalty.

Parameters
----------
standardize : bool, default: False
    Specifies if each column should be normalized

max_iter : int, default: 100
    Maximum number of iterations of the optimization algorithm.

batch_size : int, default: 1024
    Number of rows of each batch of the batch training.
    If equal to number of rows of training data, then the algorithm
    would be equivalent to full scale training.

lambda_ : float, default: 0.1
    Weight of regulerization in loss function

group_nums : dict
    a dictionary of number of features in each group, like {0:4, 1:5, 2:2, 3:6, ...}

Attributes
----------
coef_ : array, shape (1, n_features) or (n_classes, n_features)
    Coefficient of the features in the decision function.

    `coef_` is of shape (1, n_features) when the given problem
    is binary.

intercept_ : array, shape (1,)
    Intercept (a.k.a. bias) added to the decision function.

Losses : array
    Array of Loss recorded during training iterations.

Methods
-------
fit : optimize the parameters

predict_log_proba : calculate the linear predictor

predict_proba : calculate the probability of 1

predict : calculate the class

plotLoss : plot the trace plot of loss

Examples
--------
>>> from sklearn.datasets import load_iris
>>> from LinearModel import GroupLassoClassifier
>>> X, y = load_iris(return_X_y=True)
>>> model = GroupLassoClassifier(group_num = {0:5, 1:X.shape[1]-5}).fit(X, y)
>>> model.predict(X[:2, :])
array([0, 0])
>>> model.predict_proba(X[:2, :]).shape
(2, 3)
    '''
    def __init__(self,group_nums,lambda_=0.1,batch_size=1024,max_iter=100,standardize=False):
        Models.__init__(self,max_iter,batch_size,standardize)
        self.group_nums = group_nums
        self.lambda_=lambda_

    def fit(self,dataX,dataY):
        coefs, Losses = mini_Glasso(dataX,dataY,self.group_nums,self.lambda_,self.max_iter,self.batch_size,self.standardize)
        self.coef_ = coefs[:-1]
        self.intercept_ = coefs[-1]
        self.Losses = Losses


class GroupLassoClassifierCV(Models):
    '''
GroupLasso Regression classifier with l1 penalty with CV.

Parameters
----------
group_nums : dict
    a dictionary of number of features in each group, like {0:4, 1:5, 2:2, 3:6, ...}

standardize : bool, default: False
    Specifies if each column should be normalized

max_iter : int, default: 100
    Maximum number of iterations of the optimization algorithm.

batch_size : int, default: 1024
    Number of rows of each batch of the batch training.
    If equal to number of rows of training data, then the algorithm
    would be equivalent to full scale training.

cv_method : str, default: 'auc'
    criterion of CV, options include 'auc' and 'f1_score'


Attributes
----------
coef_ : array, shape (1, n_features) or (n_classes, n_features)
    Coefficient of the features in the decision function.

    `coef_` is of shape (1, n_features) when the given problem
    is binary.

intercept_ : array, shape (1,)
    Intercept (a.k.a. bias) added to the decision function.

Losses : array
    Array of Loss recorded during training iterations.

lambda_ : float
    selected weight of regularization in loss function by CV

num_valid_beta : int
    number of non-zero coefficients

Methods
-------
fit : optimize the parameters

predict_log_proba : calculate the linear predictor

predict_proba : calculate the probability of 1

predict : calculate the class

plotLoss : plot the trace plot of loss

Examples
--------
>>> from sklearn.datasets import load_iris
>>> from LinearModel import GroupLassoClassifierCV
>>> X, y = load_iris(return_X_y=True)
>>> model = GroupLassoClassifierCV(group_num = {0:5, 1:X.shape[1]-5}).fit(X, y)
>>> model.predict(X[:2, :])
array([0, 0])
>>> model.predict_proba(X[:2, :]).shape
(2, 3)
    '''
    def __init__(self,group_nums,max_iter=100,batch_size=1024,cv_method='auc',standardize=False):
        Models.__init__(self,max_iter,batch_size,standardize)
        self.cv_method=cv_method
        self.group_nums=group_nums

    def fit(self,dataX,dataY):
        coefs, num_valid_beta, lambda_, Losses = Glasso_reg(dataX,dataY,self.group_nums,self.max_iter,self.batch_size,self.cv_method,self.standardize)
        self.coef_ = coefs[:-1]
        self.intercept_ = coefs[-1]
        self.lambda_ = lambda_
        self.num_valid_beta = num_valid_beta
        self.Losses = Losses


class AlphaLassoClassifier(Models):
    '''
AlphaLasso Regression classifier with l1 penalty.
AlphaLasso only do l1 norm on specified parameters. Hence,
it would make sure that other specified parameters will be
included in the selected model.

Parameters
----------
standardize : bool, default: False
    Specifies if each column should be normalized

max_iter : int, default: 100
    Maximum number of iterations of the optimization algorithm.

batch_size : int, default: 1024
    Number of rows of each batch of the batch training.
    If equal to number of rows of training data, then the algorithm
    would be equivalent to full scale training.

lambda_ : float, default: 0.1
    Weight of regulerization in loss function

beta_index : list
    a list of index of the position for the parameters that must be
    included in the final model.

Attributes
----------
coef_ : array, shape (1, n_features) or (n_classes, n_features)
    Coefficient of the features in the decision function.

    `coef_` is of shape (1, n_features) when the given problem
    is binary.

intercept_ : array, shape (1,)
    Intercept (a.k.a. bias) added to the decision function.

Losses : array
    Array of Loss recorded during training iterations.

Methods
-------
fit : optimize the parameters

predict_log_proba : calculate the linear predictor

predict_proba : calculate the probability of 1

predict : calculate the class

plotLoss : plot the trace plot of loss

Examples
--------
>>> from sklearn.datasets import load_iris
>>> from LinearModel import AlphaLassoClassifier
>>> X, y = load_iris(return_X_y=True)
>>> model = AlphaLassoClassifier(range(3)).fit(X, y)
>>> model.predict(X[:2, :])
array([0, 0])
>>> model.predict_proba(X[:2, :]).shape
(2, 3)
    '''
    def __init__(self,beta_index,lambda_=0.1,max_iter=100,batch_size=1024,standardize=False):
        Models.__init__(self,max_iter,batch_size,standardize)
        self.beta_index=beta_index
        self.lambda_=lambda_

    def fit(self,dataX,dataY):
        self.alpha_index = [i for i in range(dataX.shape[1]) if i not in self.beta_index]
        dataX_beta = dataX[:,self.beta_index]
        dataX_alpha = dataX[:,self.alpha_index]
        coefs, Losses = mini_alpha_lasso(dataX_beta,dataX_alpha,dataY,self.lambda_,self.max_iter,self.batch_size,self.standardize)
        self.coef_ = coefs[:-1]
        self.intercept_ = coefs[-1]
        self.Losses = Losses


class AlphaLassoClassifierCV(Models):
    '''
AlphaLasso Regression classifier with l1 penalty with CV.
AlphaLasso only do l1 norm on specified parameters. Hence,
it would make sure that other specified parameters will be
included in the selected model.

Parameters
----------
beta_index : list
    a list of index of the position for the parameters that must be
    included in the final model.

standardize : bool, default: False
    Specifies if each column should be normalized

max_iter : int, default: 100
    Maximum number of iterations of the optimization algorithm.

batch_size : int, default: 1024
    Number of rows of each batch of the batch training.
    If equal to number of rows of training data, then the algorithm
    would be equivalent to full scale training.

cv_method : str, default: 'auc'
    criterion of CV, options include 'auc' and 'f1_score'


Attributes
----------
coef_ : array, shape (1, n_features) or (n_classes, n_features)
    Coefficient of the features in the decision function.

    `coef_` is of shape (1, n_features) when the given problem
    is binary.

intercept_ : array, shape (1,)
    Intercept (a.k.a. bias) added to the decision function.

Losses : array
    Array of Loss recorded during training iterations.

lambda_ : float
    selected weight of regularization in loss function by CV

num_valid_beta : int
    number of non-zero coefficients

Methods
-------
fit : optimize the parameters

predict_log_proba : calculate the linear predictor

predict_proba : calculate the probability of 1

predict : calculate the class

plotLoss : plot the trace plot of loss

Examples
--------
>>> from sklearn.datasets import load_iris
>>> from LinearModel import AlphaLassoClassifierCV
>>> X, y = load_iris(return_X_y=True)
>>> model = AlphaLassoClassifierCV(range(3)).fit(X, y)
>>> model.predict(X[:2, :])
array([0, 0])
>>> model.predict_proba(X[:2, :]).shape
(2, 3)
    '''
    def __init__(self,beta_index,max_iter=100,batch_size=1024,cv_method='auc',standardize=False):
        Models.__init__(self,max_iter,batch_size,standardize)
        self.cv_method=cv_method
        self.beta_index=[i for i in beta_index]


    def fit(self,dataX,dataY):
        coefs, num_valid_beta, lambda_, Losses = alpha_lasso_reg(dataX,dataY,self.beta_index,self.max_iter,self.batch_size,self.cv_method,self.standardize)
        self.coef_ = coefs[:-1]
        self.intercept_ = coefs[-1]
        self.lambda_ = lambda_
        self.num_valid_beta = num_valid_beta
        self.Losses = Losses




def calculate_f1(Y_test,Y_hat):
    Y_hat = pd.DataFrame(Y_hat)
    Y_hat.loc[Y_hat.index[Y_hat.loc[:,0]<0.5],0] = 0
    Y_hat.loc[Y_hat.index[Y_hat.loc[:,0]>=0.5],0] = 1
    Y_hat = np.array(Y_hat[0])
    return(f1_score(Y_test,Y_hat))


def KS_threshold(Y_test_,Y_hat_):
    Y_test = copy.deepcopy(Y_test_)
    Y_hat = copy.deepcopy(Y_hat_)
    Y_test = pd.DataFrame(Y_test)
    Y_hat = pd.DataFrame(Y_hat)
    KS = []
    for i in range(101):
        index_ = Y_hat.index[Y_hat[0]<=i/100]
        TP = np.float(sum(Y_test.loc[Y_hat[0]>i/100,:][0]))
        FN = np.float(sum(Y_test.loc[Y_hat[0]<=i/100,:][0]))
        FP = np.float(np.sum(Y_hat[0]>i/100)) - np.float(sum(Y_test.loc[Y_hat[0]>i/100,:][0]))
        TN = np.float(np.sum(Y_hat[0]<=i/100)) - np.float(sum(Y_test.loc[Y_hat[0]<=i/100,:][0]))

        if (TP+FN)==0 or (FP+TN)==0:
            KS.append(0)
        else:
            TPR = TP/(TP+FN)
            FPR = FP/(FP+TN)
            KS.append(TPR-FPR)
    return(KS)

def PlotKS(preds, labels, n=100, asc=0):
    # preds is score: asc=1
    # preds is prob: asc=0
    pred = preds # 预测值
    bad = labels # 取1为bad, 0为good
    ksds = pd.DataFrame({'bad': bad, 'pred': pred})
    ksds.loc[:,'good'] = 1 - ksds.bad

    if asc == 1:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, True])
    elif asc == 0:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, True])
    ksds1.index = range(len(ksds1.pred))
    ksds1.loc[:,'cumsum_good1'] = 1.0*ksds1.good.cumsum()/sum(ksds1.good)
    ksds1.loc[:,'cumsum_bad1'] = 1.0*ksds1.bad.cumsum()/sum(ksds1.bad)

    if asc == 1:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, False])
    elif asc == 0:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, False])
    ksds2.index = range(len(ksds2.pred))
    ksds2.loc[:,'cumsum_good2'] = 1.0*ksds2.good.cumsum()/sum(ksds2.good)
    ksds2.loc[:,'cumsum_bad2'] = 1.0*ksds2.bad.cumsum()/sum(ksds2.bad)

    # ksds1 ksds2 -> average
    ksds = ksds1.loc[:,['cumsum_good1', 'cumsum_bad1']]
    ksds.loc[:,'cumsum_good2'] = ksds2.loc[:,'cumsum_good2']
    ksds.loc[:,'cumsum_bad2'] = ksds2.loc[:,'cumsum_bad2']
    ksds.loc[:,'cumsum_good'] = (ksds.loc[:,'cumsum_good1'] + ksds.loc[:,'cumsum_good2'])/2
    ksds.loc[:,'cumsum_bad'] = (ksds.loc[:,'cumsum_bad1'] + ksds.loc[:,'cumsum_bad2'])/2

    # ks
    ksds.loc[:,'ks'] = ksds.loc[:,'cumsum_bad'] - ksds.loc[:,'cumsum_good']
    ksds.loc[:,'tile0'] = range(1, len(ksds.ks) + 1)
    ksds.loc[:,'tile'] = 1.0*ksds.loc[:,'tile0']/len(ksds.loc[:,'tile0'])

    qe = list(np.arange(0, 1, 1.0/n))
    qe.append(1)
    qe = qe[1:]

    ks_index = pd.Series(ksds.index)
    ks_index = ks_index.quantile(q = qe)
    ks_index = np.ceil(ks_index).astype(int)
    ks_index = list(ks_index)

    ksds = ksds.loc[ks_index]
    ksds = ksds[['tile', 'cumsum_good', 'cumsum_bad', 'ks']]
    ksds0 = np.array([[0, 0, 0, 0]])
    ksds = np.concatenate([ksds0, ksds], axis=0)
    ksds = pd.DataFrame(ksds, columns=['tile', 'cumsum_good', 'cumsum_bad', 'ks'])

    ks_value = ksds.ks.max()
    ks_pop = ksds.tile[ksds.ks.idxmax()]
    #print ('ks_value is ' + str(np.round(ks_value, 4)) + ' at pop = ' + str(np.round(ks_pop, 4)))

    # chart
    plt.plot(ksds.tile, ksds.cumsum_good, label='cum_good',color='blue', linestyle='-', linewidth=2)
    plt.plot(ksds.tile, ksds.cumsum_bad, label='cum_bad',color='red', linestyle='-', linewidth=2)
    plt.plot(ksds.tile, ksds.ks, label='ks',color='green', linestyle='-', linewidth=2)
    plt.axvline(ks_pop, color='gray', linestyle='--')
    plt.axhline(ks_value, color='green', linestyle='--')
    plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_good'], color='blue', linestyle='--')
    plt.axhline(ksds.loc[ksds.ks.idxmax(),'cumsum_bad'], color='red', linestyle='--')
    plt.title('KS=%s ' %np.round(ks_value, 4) +'at Pop=%s' %np.round(ks_pop, 4), fontsize=15)
    plt.show()
    return(ks_value,ks_pop)






def MultilevelVariable(df,indicator_col,X_cols):
    '''
        Author : 苏冠旭
        将indicator_col二值化，并将每一个level和X_cols中的每一列做交互项。同时增加二值化对应的常数项
        Input:
            df : 数据框
            indicator_col : 需要进行二值化的列，字符串
            X_cols : 自变量对应的列名，list
        Output:
            [0] : 处理之后的数据框
            [1] : 更新后的自变量对应的列名，list
    '''
    import copy
    df_ = pd.get_dummies(df,columns=[indicator_col])
    new_col_names = df_.columns.tolist()
    level_col_names = [i for i in df_.columns.tolist() if i.startswith(indicator_col+'_')]
    X_cols_new = copy.deepcopy(X_cols)
    for each_level in level_col_names:
        X_cols_new.append(each_level)
        for each_X in X_cols:
            tmp_new_name = each_X + '_' + each_level.split(indicator_col+'_')[1]
            X_cols_new.append(tmp_new_name)
            df_[tmp_new_name] = np.multiply(df_[each_X],df_[each_level])
    return(df_,X_cols_new)




def MultilevelCoef(coefs,intercept,new_X_cols,level_names,indicator_col):
    '''
        Author : 苏冠旭
        和MultilevelVariable配套使用，将Multilevel Model得到的系数还原到各个行业中
        Input:
            coefs : 模型得到的系数，不包括常数项
            intercept : 模型得到的常数项
            new_X_cols : MultilevelVariable返回的feature对应的列名，list
            level_names : 分维变量的各个维度的名称组成的list，比如list(set(df[indicator_col]))
            indicator_col : 原数据框中分维度的列对应的列名，与MultilevelVariable中的indicator_col相同，字符串类型
        Output:
            一个数据框，列为各个行业，行为各个系数和常数项
    '''
    coefs = list(coefs)
    coefs.append(intercept)
    new_X_cols_ = copy.deepcopy(new_X_cols)
    new_X_cols_.append(indicator_col)
    coefs = pd.DataFrame(coefs)
    coefs.index = new_X_cols_

    X_cols_origin = []
    for each_X in new_X_cols_:
        _ = 0
        for each_level in level_names:
            if each_level in each_X:
                break
            _ += 1
        if _ == len(level_names):
            X_cols_origin.append(each_X)

    coef_by_level = {}
    for each_level in level_names:
        coef_by_level[each_level] = {}
        for each_X in X_cols_origin:
            coef_by_level[each_level][each_X] = coefs.loc[each_X,0]
            coef_by_level[each_level][each_X] += coefs.loc[each_X+'_'+each_level,0]

    coef_by_level = pd.DataFrame(coef_by_level)
    all_the_same = ['Yes' for i in range(len(coef_by_level))]
    row_num = 0
    for i in coef_by_level.index:
        _ = coef_by_level.loc[i,coef_by_level.columns[0]]
        for j in coef_by_level.columns:
            if coef_by_level.loc[i,j] != _:
                all_the_same[row_num] = 'No'
                break
        row_num+=1

    coef_by_level.index = list(coef_by_level.index)[:-1] + ['intercept']
    coef_by_level['all_the_same'] = all_the_same
    coef_by_level = coef_by_level.loc[np.sum(np.abs(coef_by_level.iloc[:,:-1]),axis=1)!=0,:]


    return(coef_by_level)




def byIndustryModel(df,X_cols,y_col,industry_col,test_size,model):
    '''
        Author : 苏冠旭
        将indicator_col二值化，并将每一个level和X_cols中的每一列做交互项。同时增加二值化对应的常数项。
        以此来建立Lasso模型，进行变量筛选。返回建模结果。

        Input :
            df : 数据框
            X_cols : 自变量对应的列名，list
            y_col : 因变量对应的列名，字符串
            industry_col : 需要进行二值化的列，字符串
            test_size : 测试集比例，float
            model : 一个类sklearn模型，包含model.fit()和model.predict()方法，包含model.coef_和model.intercept_属性
        Output :
            [0] : 一个数据框，列为各个行业，行为各个系数和常数项，由Lasso返回结果加工得到
            [1] : AUC
            [2] : 拟合之后的model
            [3] : 测试集自变量矩阵
            [4] : 测试集因变量矩阵
            [5] : 训练集自变量矩阵
            [6] : 训练集因变量矩阵
            [7] : 自变量矩阵对应的列名
            [8] : 训练集行业分类标签
            [9] : 测试集行业分类标签
    '''
    df2,new_X_cols = MultilevelVariable(df,industry_col,X_cols)
    df2.index = range(len(df2))
    tag_ = df[industry_col].values
    #tag_.index = range(len(tag_))
    random.seed(0)
    X_data = np.array(df2.loc[:,new_X_cols])
    Y_data = np.array(df2.loc[:,y_col])
    index_black = [i for i in list(df2.index) if Y_data[i]==1]
    index_test = list(df2.sample(int(test_size*len(df2)),random_state=0).index)
    index_train = list(set(list(df2.index)) - set(index_test))
    index_train = list(set(index_train) | set(index_black))
    X_train = X_data[index_train]
    X_test = X_data[index_test]
    y_train = Y_data[index_train]
    y_test = Y_data[index_test]
    industry_train = tag_[index_train]
    industry_test = tag_[index_test]
    del X_data,Y_data,df2
    model.fit(X_train, y_train)
    predicted = model.predict_proba(X_test)
    auc_ = roc_auc_score(y_test,predicted)
    X_ = MultilevelCoef(model.coef_,model.intercept_,new_X_cols,level_names=list(set(df[industry_col])),indicator_col=industry_col)
    return(X_,auc_,model,X_test,y_test,X_train,y_train,new_X_cols,industry_train,industry_test)



########################### Model Calculation##############################

def make_positive(input_tensor):
    #out_ = input_tensor*tf.cast(input_tensor>0, tf.float32)
    out_ = tf.clip_by_value(input_tensor,0.0,9999)
    return out_

def mini_PosLasso(dataX_,dataY_,lambd,pos_penalty,max_iter,batch_size=1024,standardize=False,\
                  y_track=False,beta_track=False,start_point='ridge',Q1=1,Q0=0.98):
    '''
    This function requires feature matrix, a column of label from training data,
    hyper varaible lambd, and maxinum number of iteration max_iter.
    The function give out the fitted coefficients of Positive Lasso as well as trace of loss function.
    The Positive Lasso would standardize the input data automatically if standardize=True!

    The optimazation would stop if loss function reduces less than 0.01%, or number of iterations reaches
    max_iter.
    '''
    l1_ratio = pos_penalty
    dataX = copy.deepcopy(dataX_)
    dataY = copy.deepcopy(dataY_)
    if standardize:
        std_ = []
        mean_ = []
        meanY = 0
        stdY = 1
        dataY = (dataY-meanY)/stdY
        for i in range(dataX.shape[1]):
            std_.append(np.std(dataX[:,i]))
            mean_.append(np.mean(dataX[:,i]))
            dataX[:,i] = (dataX[:,i]-mean_[i])/std_[i]
    tf.reset_default_graph()
    x = tf.placeholder(dtype = tf.float32, shape=[None,dataX.shape[1]])
    y = tf.placeholder(dtype = tf.float32, shape=[None,1])
    l = tf.constant(lambd,dtype=tf.float32)

    if start_point=='ridge':
        model_OLS = linear_model.Ridge()
        model_OLS.fit(dataX, dataY)
        coef_start = model_OLS.coef_
        beta = tf.Variable(np.reshape(coef_start,(max(coef_start.shape),1)), name='beta', dtype=tf.float32,\
            constraint=make_positive)
    else:
        beta = tf.Variable(tf.random_normal([dataX.shape[1],1], stddev=0.1,mean=0.2),name = 'beta',\
                           dtype=tf.float32,constraint=make_positive)

    if standardize:
        y_hat_linear = tf.matmul(x,beta)
    else:
        intercept = tf.Variable(tf.zeros([1,1]),name = 'intercept',dtype=tf.float32)
        y_hat_linear = tf.matmul(x,beta) + intercept
    y_hat = tf.sigmoid(y_hat_linear)
    loss = 10*tf.reduce_mean(-y*tf.log(tf.clip_by_value(y_hat,1e-10,1.0))-(1-y)*tf.log(tf.clip_by_value(1-y_hat,1e-10,1.0)))
    loss_pointwise = -y*tf.log(tf.clip_by_value(y_hat,1e-10,1.0))-(1-y)*tf.log(tf.clip_by_value(1-y_hat,1e-10,1.0))
    #loss = tf.reduce_mean(tf.losses.mean_squared_error(y,y_hat_linear))

    l1_loss = tf.reduce_sum(tf.abs(beta))

    negLoss = tf.reduce_mean(-tf.clip_by_value(beta,-9999,0))
    #loss = loss + lambd*l1_loss + l1_ratio*negLoss
    loss = loss + lambd*l1_loss
    #loss = loss + l1_ratio*negLoss

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.5,global_step,decay_steps=max_iter/10,decay_rate=0.95)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init_op)
    Loss = []
    Y_HAT = []
    beta_HAT = []
    index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]
    batch_X = dataX[index_batch]
    batch_Y = dataY[index_batch]

    sess.run(train_step,feed_dict={x:batch_X,y:np.reshape(batch_Y,(batch_Y.shape[0],1))})
    Loss.append(sess.run(loss,feed_dict={x:dataX,y:np.reshape(dataY,(dataY.shape[0],1))}))
    sess.run(train_step,feed_dict={x:batch_X,y:np.reshape(batch_Y,(batch_Y.shape[0],1))})
    Loss.append(sess.run(loss,feed_dict={x:dataX,y:np.reshape(dataY,(dataY.shape[0],1))}))
    step_ = 2

    for i in range(20):
        step_ += 1
        index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]
        batch_X = dataX[index_batch]
        batch_Y = dataY[index_batch]
        sess.run(train_step,feed_dict={x:batch_X,y:np.reshape(batch_Y,(batch_Y.shape[0],1))})
        Loss.append(sess.run(loss,feed_dict={x:dataX,y:np.reshape(dataY,(dataY.shape[0],1))}))

    while (((Loss[-2]-Loss[-1])/np.abs(Loss[-2])>.0001) or ((Loss[-2]-Loss[-1])<0) or step_<20) & (step_<=max_iter):
        step_ += 1
        index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]
        batch_X = dataX[index_batch]
        batch_Y = dataY[index_batch]

        if Q1<1 or Q0<1:
            batch_X_white = batch_X[batch_Y==0]
            batch_X_black = batch_X[batch_Y==1]
            batch_Y_white = batch_Y[batch_Y==0]
            batch_Y_black = batch_Y[batch_Y==1]

            Loss_white = sess.run(loss_pointwise,feed_dict={x:batch_X_white,y:np.reshape(batch_Y_white,(batch_Y_white.shape[0],1))})
            Loss_black = sess.run(loss_pointwise,feed_dict={x:batch_X_black,y:np.reshape(batch_Y_black,(batch_Y_black.shape[0],1))})
            Loss_white = np.reshape(Loss_white,max(Loss_white.shape))
            Loss_black = np.reshape(Loss_black,max(Loss_black.shape))

            white_NoOutlier = np.array(Loss_white<=np.quantile(Loss_white,Q0))
            black_NoOutlier = np.array(Loss_black<=np.quantile(Loss_black,Q1))

            batch_X = np.concatenate((batch_X_white[white_NoOutlier],batch_X_black[black_NoOutlier]))
            batch_Y = np.concatenate((batch_Y_white[white_NoOutlier],batch_Y_black[black_NoOutlier]))

        sess.run(train_step,feed_dict={x:batch_X,y:np.reshape(batch_Y,(batch_Y.shape[0],1))})
        Loss.append(sess.run(loss,feed_dict={x:dataX,y:np.reshape(dataY,(dataY.shape[0],1))}))
        if step_ % 20 == 0:
            time_ = datetime.datetime.now().strftime('%Y-%D %H:%M:%S')
            print(time_ + '\t'+str(step_) +'\t'+ str(Loss[-1]))

    if y_track or beta_track:
        for i in range(100):
            index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]
            batch_X = dataX[index_batch]
            batch_Y = dataY[index_batch]

            if Q1<1 or Q0<1:
                batch_X_white = batch_X[batch_Y==0]
                batch_X_black = batch_X[batch_Y==1]
                batch_Y_white = batch_Y[batch_Y==0]
                batch_Y_black = batch_Y[batch_Y==1]

                Loss_white = sess.run(loss_pointwise,feed_dict={x:batch_X_white,y:np.reshape(batch_Y_white,(batch_Y_white.shape[0],1))})
                Loss_black = sess.run(loss_pointwise,feed_dict={x:batch_X_black,y:np.reshape(batch_Y_black,(batch_Y_black.shape[0],1))})
                Loss_white = np.reshape(Loss_white,max(Loss_white.shape))
                Loss_black = np.reshape(Loss_black,max(Loss_black.shape))

                white_NoOutlier = np.array(Loss_white<=np.quantile(Loss_white,Q0))
                black_NoOutlier = np.array(Loss_black<=np.quantile(Loss_black,Q1))

                batch_X = np.concatenate((batch_X_white[white_NoOutlier],batch_X_black[black_NoOutlier]))
                batch_Y = np.concatenate((batch_Y_white[white_NoOutlier],batch_Y_black[black_NoOutlier]))

            sess.run(train_step,feed_dict={x:batch_X,y:np.reshape(batch_Y,(batch_Y.shape[0],1))})
            Loss.append(sess.run(loss,feed_dict={x:dataX,y:np.reshape(dataY,(dataY.shape[0],1))}))
            beta_HAT.append(sess.run(beta))
            Y_HAT.append(sess.run(y_hat,feed_dict={x:dataX,y:np.reshape(dataY,(dataY.shape[0],1))}))

        beta_HAT = np.array(beta_HAT)
        beta_HAT = np.reshape(beta_HAT,beta_HAT.shape[:2]).T

    if standardize:
        coefs = np.zeros((dataX.shape[1]+1))
        betas = sess.run(beta)
        a = pd.DataFrame(betas)
        a.loc[a.index[a.loc[:,0].values<=0.001],0] = 0
        betas = a[0]
        for i in range(dataX.shape[1]):
            coefs[i] = stdY/std_[i]*betas[i]
            coefs[-1] -= mean_[i]/std_[i]*betas[i]*stdY
        coefs[-1] += meanY
    else:
        coefs = np.zeros((dataX.shape[1]+1))
        betas = sess.run(beta)
        a = pd.DataFrame(betas)
        a.loc[a.index[a.loc[:,0].values<=0.001],0] = 0
        coefs[:-1] = a[0]
        coefs[-1] = sess.run(intercept)
    tf.reset_default_graph()
    return(coefs,Loss,Y_HAT,beta_HAT)

def PosLasso_reg(dataX,dataY,max_iter,batch_size,cv_method = 'auc',standardize=False,start_point='ridge',Q1=1,Q0=0.98):
    lambd = [1e-2,1e-3,1e-4,1e-5,1e-6]
    pos_penalty = [1]

    score = []
    for l in lambd:
        score.append([])
        for l1 in pos_penalty:
            score_ = 0
            for _ in range(2):
                print('CV loop No.'+str(_)+' for lambda='+str(l))
                X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.33, random_state=42)
                coef_ = mini_PosLasso(X_train,Y_train,l,l1,max_iter,batch_size=batch_size,\
                                      standardize=standardize,start_point=start_point,Q1=Q1,Q0=Q0)[0]
                Y_hat = np.matmul(np.append(X_test,np.ones((len(X_test),1)),axis=1),coef_)
                if cv_method=='auc':
                    try:
                        score_ += roc_auc_score(Y_test,Y_hat)
                    except:
                        score_ += 0
                if cv_method=='f1_score':
                    Y_hat = pd.DataFrame(Y_hat)
                    Y_hat.loc[Y_hat.index[Y_hat.loc[:,0]<0.5],0] = 0
                    Y_hat.loc[Y_hat.index[Y_hat.loc[:,0]>=0.5],0] = 1
                    Y_hat = np.array(Y_hat[0])
                    try:
                        score_ += calculate_f1(Y_test,Y_hat)
                    except:
                        score_ += 0
            score[-1].append(score_/3)
    score = np.array(score)
    lambd_ = 0
    l1_ratio_ = 0
    for i in range(len(lambd)):
        for j in range(len(pos_penalty)):
            if score[i,j] == np.max(score):
                lambd_ = lambd[i]
                l1_ratio_ = pos_penalty[j]
    print('Set lambda='+str(lambd_)+' and fit model...')
    coefs, Loss, Y_HAT, beta_HAT = mini_PosLasso(dataX,dataY,lambd_,l1_ratio_,max_iter,y_track=True,\
                                                 standardize=standardize,start_point=start_point,Q1=Q1,Q0=Q0)
    print("Positive Lasso  picked " + str(sum(coefs != 0)-1) + \
          " variables and eliminated the other " +  str(sum(coefs == 0)) + " variables")
    return(coefs,sum(coefs != 0)-1,lambd_,l1_ratio_,Loss,Y_HAT,beta_HAT)




def mini_enet(dataX_,dataY_,lambd,l1_ratio,max_iter,batch_size=1024,standardize=False,y_track=False,\
              beta_track=False,start_point='ridge',Q1=1,Q0=0.98):
    '''
    此函数用于拟合

    This function requires feature matrix, a column of label from training data,
    hyper varaible lambd, and maxinum number of iteration max_iter.
    The function give out the fitted coefficients of ElasticNet as well as trace of loss function.
    The ElasticNet would standardize the input data automatically!

    The optimazation would stop if loss function reduces less than 0.01%, or number of iterations reaches
    max_iter.
    '''
    dataX = copy.deepcopy(dataX_)
    dataY = copy.deepcopy(dataY_)
    if standardize:
        std_ = []
        mean_ = []
        meanY = np.mean(dataY)
        stdY = np.std(dataY)
        dataY = (dataY-meanY)/stdY
        for i in range(dataX.shape[1]):
            std_.append(np.std(dataX[:,i]))
            mean_.append(np.mean(dataX[:,i]))
            dataX[:,i] = (dataX[:,i]-mean_[i])/std_[i]
    tf.reset_default_graph()
    x = tf.placeholder(dtype = tf.float32, shape=[None,dataX.shape[1]])
    y = tf.placeholder(dtype = tf.float32, shape=[None,1])
    l = tf.constant(lambd,dtype=tf.float32)
    if start_point=='ridge':
        model_OLS = linear_model.Ridge()
        model_OLS.fit(dataX, dataY)
        coef_start = model_OLS.coef_
        beta = tf.Variable(np.reshape(coef_start,(max(coef_start.shape),1)), name='beta', dtype=tf.float32)
    else:
        beta = tf.Variable(tf.random_normal([dataX.shape[1],1], stddev=0.1,mean=0.2),name = 'beta',dtype=tf.float32)
    if standardize:
        y_hat_linear = tf.matmul(x,beta)
    else:
        intercept = tf.Variable(tf.zeros([1,1]),name = 'intercept',dtype=tf.float32)
        y_hat_linear = tf.matmul(x,beta) + intercept
    y_hat = tf.sigmoid(y_hat_linear)
    loss = 10*tf.reduce_mean(-y*tf.log(tf.clip_by_value(y_hat,1e-10,1.0))-(1-y)*tf.log(tf.clip_by_value(1-y_hat,1e-10,1.0)))
    loss_pointwise = -y*tf.log(tf.clip_by_value(y_hat,1e-10,1.0))-(1-y)*tf.log(tf.clip_by_value(1-y_hat,1e-10,1.0))

    l1_loss = tf.reduce_sum(tf.abs(beta))
    l2_loss = tf.reduce_sum(tf.square(beta))
    loss = loss + lambd*l1_ratio*l1_loss + lambd*(1-l1_ratio)/2*l2_loss
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.5,global_step,decay_steps=max_iter/10,decay_rate=0.95)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init_op)
    Loss = []
    Y_HAT = []
    beta_HAT = []
    index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]
    batch_X = dataX[index_batch]
    batch_Y = dataY[index_batch]

    sess.run(train_step,feed_dict={x:batch_X,y:np.reshape(batch_Y,(batch_Y.shape[0],1))})
    Loss.append(sess.run(loss,feed_dict={x:dataX,y:np.reshape(dataY,(dataY.shape[0],1))}))
    sess.run(train_step,feed_dict={x:batch_X,y:np.reshape(batch_Y,(batch_Y.shape[0],1))})
    Loss.append(sess.run(loss,feed_dict={x:dataX,y:np.reshape(dataY,(dataY.shape[0],1))}))
    step_ = 2

    for i in range(20):
        step_ += 1
        index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]
        batch_X = dataX[index_batch]
        batch_Y = dataY[index_batch]
        sess.run(train_step,feed_dict={x:batch_X,y:np.reshape(batch_Y,(batch_Y.shape[0],1))})
        Loss.append(sess.run(loss,feed_dict={x:dataX,y:np.reshape(dataY,(dataY.shape[0],1))}))

    while (((Loss[-2]-Loss[-1])/np.abs(Loss[-2])>.0001) or ((Loss[-2]-Loss[-1])<0) or step_<20) & (step_<=max_iter):
        step_ += 1
        index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]
        batch_X = dataX[index_batch]
        batch_Y = dataY[index_batch]

        if Q1<1 or Q0<1:
            batch_X_white = batch_X[batch_Y==0]
            batch_X_black = batch_X[batch_Y==1]
            batch_Y_white = batch_Y[batch_Y==0]
            batch_Y_black = batch_Y[batch_Y==1]

            Loss_white = sess.run(loss_pointwise,feed_dict={x:batch_X_white,y:np.reshape(batch_Y_white,(batch_Y_white.shape[0],1))})
            Loss_black = sess.run(loss_pointwise,feed_dict={x:batch_X_black,y:np.reshape(batch_Y_black,(batch_Y_black.shape[0],1))})
            Loss_white = np.reshape(Loss_white,max(Loss_white.shape))
            Loss_black = np.reshape(Loss_black,max(Loss_black.shape))

            white_NoOutlier = np.array(Loss_white<=np.quantile(Loss_white,Q0))
            black_NoOutlier = np.array(Loss_black<=np.quantile(Loss_black,Q1))

            batch_X = np.concatenate((batch_X_white[white_NoOutlier],batch_X_black[black_NoOutlier]))
            batch_Y = np.concatenate((batch_Y_white[white_NoOutlier],batch_Y_black[black_NoOutlier]))

        sess.run(train_step,feed_dict={x:batch_X,y:np.reshape(batch_Y,(batch_Y.shape[0],1))})
        Loss.append(sess.run(loss,feed_dict={x:dataX,y:np.reshape(dataY,(dataY.shape[0],1))}))
        if step_ % 10 == 0:
            time_ = datetime.datetime.now().strftime('%Y-%D %H:%M:%S')
            print(time_ + '\t'+str(step_) +'\t'+ str(Loss[-1]))

    if y_track or beta_track:
        for i in range(100):
            index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]
            batch_X = dataX[index_batch]
            batch_Y = dataY[index_batch]

            if Q1<1 or Q0<1:
                batch_X_white = batch_X[batch_Y==0]
                batch_X_black = batch_X[batch_Y==1]
                batch_Y_white = batch_Y[batch_Y==0]
                batch_Y_black = batch_Y[batch_Y==1]

                Loss_white = sess.run(loss_pointwise,feed_dict={x:batch_X_white,y:np.reshape(batch_Y_white,(batch_Y_white.shape[0],1))})
                Loss_black = sess.run(loss_pointwise,feed_dict={x:batch_X_black,y:np.reshape(batch_Y_black,(batch_Y_black.shape[0],1))})
                Loss_white = np.reshape(Loss_white,max(Loss_white.shape))
                Loss_black = np.reshape(Loss_black,max(Loss_black.shape))

                white_NoOutlier = np.array(Loss_white<=np.quantile(Loss_white,Q0))
                black_NoOutlier = np.array(Loss_black<=np.quantile(Loss_black,Q1))

                batch_X = np.concatenate((batch_X_white[white_NoOutlier],batch_X_black[black_NoOutlier]))
                batch_Y = np.concatenate((batch_Y_white[white_NoOutlier],batch_Y_black[black_NoOutlier]))

            sess.run(train_step,feed_dict={x:batch_X,y:np.reshape(batch_Y,(batch_Y.shape[0],1))})
            Loss.append(sess.run(loss,feed_dict={x:dataX,y:np.reshape(dataY,(dataY.shape[0],1))}))
            beta_HAT.append(sess.run(beta))
            Y_HAT.append(sess.run(y_hat,feed_dict={x:dataX,y:np.reshape(dataY,(dataY.shape[0],1))}))

        beta_HAT = np.array(beta_HAT)
        beta_HAT = np.reshape(beta_HAT,beta_HAT.shape[:2]).T

    if standardize:
        coefs = np.zeros((dataX.shape[1]+1))
        betas = sess.run(beta)
        a = pd.DataFrame(betas)
        a.loc[a.index[np.abs(a.loc[:,0])<=0.01],0] = 0
        betas = a[0]
        for i in range(dataX.shape[1]):
            coefs[i] = stdY/std_[i]*betas[i]
            coefs[-1] -= mean_[i]/std_[i]*betas[i]*stdY
        coefs[-1] += meanY
    else:
        coefs = np.zeros((dataX.shape[1]+1))
        betas = sess.run(beta)
        a = pd.DataFrame(betas)
        a.loc[a.index[np.abs(a.loc[:,0])<=0.01],0] = 0
        coefs[:-1] = a[0]
        coefs[-1] = sess.run(intercept)
    tf.reset_default_graph()
    return(coefs,Loss,Y_HAT,beta_HAT)

def enet_reg(dataX,dataY,max_iter,batch_size,cv_method = 'auc',standardize=False,start_point='ridge',Q1=1,Q0=0.98):
    lambd = [1, 0.5, 0.1, 0.05, 0.001,0.0001]
    l1_ratio = [.01, .1, .5, .9, .99]
    score = []
    for l in lambd:
        score.append([])
        for l1 in l1_ratio:
            score_ = 0
            for _ in range(3):
                print('CV loop No.'+str(_)+' for lambda='+str(l)+' and l1_ratio='+str(l1))
                X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.33, random_state=42)
                coef_ = mini_enet(X_train,Y_train,l,l1,max_iter,batch_size=batch_size,\
                                  standardize=standardize,start_point=start_point,Q1=Q1,Q0=Q0)[0]
                Y_hat = np.matmul(np.append(X_test,np.ones((len(X_test),1)),axis=1),coef_)
                if cv_method=='auc':
                    try:
                        score_ += roc_auc_score(Y_test,Y_hat)
                    except:
                        score_ += 0
                if cv_method=='f1_score':
                    Y_hat = pd.DataFrame(Y_hat)
                    Y_hat.loc[Y_hat.index[Y_hat.loc[:,0]<0.5],0] = 0
                    Y_hat.loc[Y_hat.index[Y_hat.loc[:,0]>=0.5],0] = 1
                    Y_hat = np.array(Y_hat[0])
                    try:
                        score_ += calculate_f1(Y_test,Y_hat)
                    except:
                        score_ += 0
            score[-1].append(score_/3)
    score = np.array(score)
    lambd_ = 0
    l1_ratio_ = 0
    for i in range(len(lambd)):
        for j in range(len(l1_ratio)):
            if score[i,j] == np.max(score):
                lambd_ = lambd[i]
                l1_ratio_ = l1_ratio[j]
    print('Set lambda='+str(lambd_)+' and l1_ratio='+str(l1_ratio_)+' and fit model...')
    coefs, Loss, Y_HAT, beta_HAT = mini_enet(dataX,dataY,lambd_,l1_ratio_,max_iter,y_track=True,\
                                             standardize=standardize,start_point=start_point,Q1=Q1,Q0=Q0)
    print("ElasticNet picked " + str(sum(coefs != 0)-1) + \
          " variables and eliminated the other " +  str(sum(coefs == 0)) + " variables")
    return(coefs,sum(coefs != 0)-1,lambd_,l1_ratio_,Loss,Y_HAT,beta_HAT)


def mini_alpha_lasso(dataX_beta,dataX_alpha,dataY_,lambd,max_iter,batch_size=1024,standardize=False):
    '''
    This function requires feature matrix, a column of label from training data,
    hyper varaible lambd, and maxinum number of iteration max_iter.
    The function give out the fitted coefficients of lasso as well as trace of loss function.
    The lasso would standardize the input data automatically!

    The optimazation would stop if loss function reduces less than 0.01%, or number of iterations reaches
    max_iter.
    '''
    dataX = dataX_beta
    dataY = dataY_
    tf.reset_default_graph()
    x_beta = tf.placeholder(dtype = tf.float32, shape=[None,dataX.shape[1]])
    x_alpha = tf.placeholder(dtype = tf.float32, shape=[None,dataX_alpha.shape[1]])
    y = tf.placeholder(dtype = tf.float32, shape=[None,1])
    l = tf.constant(lambd,dtype=tf.float32)
    beta = tf.Variable(tf.zeros([dataX.shape[1],1]),name = 'beta',dtype=tf.float32)
    alpha = tf.Variable(tf.zeros([dataX_alpha.shape[1],1]),name = 'beta',dtype=tf.float32)
    intercept = tf.Variable(tf.zeros([1,1]),name = 'intercept',dtype=tf.float32)
    y_hat_linear = tf.matmul(x_beta,beta) + tf.matmul(x_alpha,alpha) + intercept
    y_hat = tf.sigmoid(y_hat_linear)
    loss = 10*tf.reduce_mean(-y*tf.log(tf.clip_by_value(y_hat,1e-10,1.0))-(1-y)*tf.log(tf.clip_by_value(1-y_hat,1e-10,1.0)))
    loss = loss + lambd*tf.reduce_sum(tf.abs(alpha)) + lambd/10*tf.reduce_sum(tf.square(beta))
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.5,global_step,decay_steps=max_iter/10,decay_rate=0.95)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init_op)
    Loss = []
    index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]
    batch_X = dataX[index_batch]
    batch_X_alpha = dataX_alpha[index_batch]
    batch_Y = dataY[index_batch]
    sess.run(train_step,feed_dict={x_beta:batch_X,x_alpha:batch_X_alpha,y:np.reshape(batch_Y,(batch_Y.shape[0],1))})
    Loss.append(sess.run(loss,feed_dict={x_beta:dataX,x_alpha:dataX_alpha,y:np.reshape(dataY,(dataY.shape[0],1))}))
    sess.run(train_step,feed_dict={x_beta:batch_X,x_alpha:batch_X_alpha,y:np.reshape(batch_Y,(batch_Y.shape[0],1))})
    Loss.append(sess.run(loss,feed_dict={x_beta:dataX,x_alpha:dataX_alpha,y:np.reshape(dataY,(dataY.shape[0],1))}))
    step_ = 2
    while (((Loss[-2]-Loss[-1])/np.abs(Loss[-2])>.0001) or ((Loss[-2]-Loss[-1])<0) or step_<30) & (step_<=max_iter):
        step_ += 1
        index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]
        batch_X = dataX[index_batch]
        batch_X_alpha = dataX_alpha[index_batch]
        batch_Y = dataY[index_batch]
        sess.run(train_step,feed_dict={x_beta:batch_X,x_alpha:batch_X_alpha,y:np.reshape(batch_Y,(batch_Y.shape[0],1))})
        Loss.append(sess.run(loss,feed_dict={x_beta:dataX,x_alpha:dataX_alpha,y:np.reshape(dataY,(dataY.shape[0],1))}))
        if step_ % 10 == 0:
            time_ = datetime.datetime.now().strftime('%Y-%D %H:%M:%S')
            print(time_ + '\t'+str(step_) +'\t'+ str(Loss[-1]))

    coefs = np.zeros((dataX.shape[1]+dataX_alpha.shape[1]+1))
    betas = sess.run(beta)
    alphas = sess.run(alpha)
    a = pd.DataFrame(alphas)
    a.loc[a.index[np.abs(a.loc[:,0])<=0.01],0] = 0
    b = pd.DataFrame(betas)
    #b.loc[b.index[np.abs(b.loc[:,0])<=0.2],0] = 0
    coefs[:len(betas)] = b[0]
    coefs[len(betas):-1] = a[0]
    coefs[-1] = sess.run(intercept)
    tf.reset_default_graph()
    return(coefs,Loss)


def alpha_lasso_reg(dataX,dataY,beta_index,max_iter,batch_size,cv_method = 'auc',standardize=False):
    lambd = [0.5, 0.1, 0.01, 0.0001]
    score = []
    alpha_index = [i for i in range(dataX.shape[1]) if i not in beta_index]
    for l in lambd:
        score_ = 0
        for _ in range(1):
            X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.33, random_state=42)
            X_train_beta = X_train[:,beta_index]
            X_test_beta = X_test[:,beta_index]
            X_train_alpha = X_train[:,alpha_index]
            X_test_alpha = X_test[:,alpha_index]
            coef_ = mini_alpha_lasso(X_train_beta,X_train_alpha,Y_train,l,max_iter=max_iter,batch_size=batch_size)[0]
            Y_hat = np.matmul(np.append(np.append(X_test_beta,X_test_alpha,axis=1),np.ones((len(X_test),1)),axis=1),coef_)
            if cv_method=='auc':
                try:
                    score_ += roc_auc_score(Y_test,Y_hat)
                except:
                    score_ += 0
            if cv_method=='f1_score':
                Y_hat = pd.DataFrame(Y_hat)
                Y_hat.loc[Y_hat.index[Y_hat.loc[:,0]<0.5],0] = 0
                Y_hat.loc[Y_hat.index[Y_hat.loc[:,0]>=0.5],0] = 1
                Y_hat = np.array(Y_hat[0])
                try:
                    score_ += calculate_f1(Y_test,Y_hat)
                except:
                    score_ += 0
        score.append(score_/3)
    lambd_ = lambd[[i for i in range(len(score)) if score[i] == max(score)][0]]
    coefs,Loss = mini_alpha_lasso(dataX[:,beta_index],dataX[:,alpha_index],dataY,lambd_,max_iter)
    print("Lasso picked " + str(sum(coefs != 0)-1) + \
          " variables and eliminated the other " +  str(sum(coefs == 0)) + " variables")
    return(coefs,sum(coefs != 0)-1,lambd_,Loss)






def mini_lasso(dataX_,dataY_,lambd,max_iter,batch_size=1024,standardize=False,y_track=False,beta_track=False,\
               start_point='ridge',Q1=1,Q0=0.98):
    '''
    This function requires feature matrix, a column of label from training data,
    hyper varaible lambd, and maxinum number of iteration max_iter.
    The function give out the fitted coefficients of lasso as well as trace of loss function.
    The lasso would standardize the input data automatically!

    The optimazation would stop if loss function reduces less than 0.01%, or number of iterations reaches
    max_iter.
    '''
    dataX = copy.deepcopy(dataX_)
    dataY = copy.deepcopy(dataY_)
    if standardize:
        std_ = []
        mean_ = []
        meanY = 0
        stdY = 1
        dataY = (dataY-meanY)/stdY
        for i in range(dataX.shape[1]):
            std_.append(np.std(dataX[:,i]))
            mean_.append(np.mean(dataX[:,i]))
            dataX[:,i] = (dataX[:,i]-mean_[i])/std_[i]
    tf.reset_default_graph()
    x = tf.placeholder(dtype = tf.float32, shape=[None,dataX.shape[1]])
    y = tf.placeholder(dtype = tf.float32, shape=[None,1])
    l = tf.constant(lambd,dtype=tf.float32)
    if start_point=='ridge':
        model_OLS = linear_model.Ridge()
        model_OLS.fit(dataX, dataY)
        coef_start = model_OLS.coef_
        beta = tf.Variable(np.reshape(coef_start,(max(coef_start.shape),1)), name='beta', dtype=tf.float32)
    else:
        beta = tf.Variable(tf.random_normal([dataX.shape[1],1], stddev=0.1,mean=0.2),name = 'beta',dtype=tf.float32)
    if standardize:
        y_hat_linear = tf.matmul(x,beta)
    else:
        intercept = tf.Variable(tf.zeros([1,1]),name = 'intercept',dtype=tf.float32)
        y_hat_linear = tf.matmul(x,beta) + intercept
    y_hat = tf.sigmoid(y_hat_linear)
    #loss = 10*tf.reduce_mean(-y*tf.log(tf.clip_by_value(y_hat,1e-10,1.0))-(1-y)*tf.log(tf.clip_by_value(1-y_hat,1e-10,1.0)))
    loss = 10*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = y_hat_linear)) + lambd*tf.reduce_sum(tf.abs(beta))
    loss_pointwise = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_hat_linear)

    #loss = loss + lambd*tf.reduce_sum(tf.abs(beta))
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.5,global_step,decay_steps=max_iter/10,decay_rate=0.95)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init_op)
    Loss = []
    Y_HAT = []
    beta_HAT = []
    index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]
    batch_X = dataX[index_batch]
    batch_Y = dataY[index_batch]
    sess.run(train_step,feed_dict={x:batch_X,y:np.reshape(batch_Y,(batch_Y.shape[0],1))})
    Loss.append(sess.run(loss,feed_dict={x:dataX,y:np.reshape(dataY,(dataY.shape[0],1))}))
    sess.run(train_step,feed_dict={x:batch_X,y:np.reshape(batch_Y,(batch_Y.shape[0],1))})
    Loss.append(sess.run(loss,feed_dict={x:dataX,y:np.reshape(dataY,(dataY.shape[0],1))}))
    step_ = 2

    for i in range(20):
        step_ += 1
        index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]
        batch_X = dataX[index_batch]
        batch_Y = dataY[index_batch]
        sess.run(train_step,feed_dict={x:batch_X,y:np.reshape(batch_Y,(batch_Y.shape[0],1))})
        Loss.append(sess.run(loss,feed_dict={x:dataX,y:np.reshape(dataY,(dataY.shape[0],1))}))

    while (((Loss[-2]-Loss[-1])/(np.abs(Loss[-2])+1e-10)>.0001) or ((Loss[-2]-Loss[-1])<0) or step_<20) & (step_<=max_iter):
        step_ += 1
        index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]
        batch_X = dataX[index_batch]
        batch_Y = dataY[index_batch]

        if Q1<1 or Q0<1:
            batch_X_white = batch_X[batch_Y==0]
            batch_X_black = batch_X[batch_Y==1]
            batch_Y_white = batch_Y[batch_Y==0]
            batch_Y_black = batch_Y[batch_Y==1]

            Loss_white = sess.run(loss_pointwise,feed_dict={x:batch_X_white,y:np.reshape(batch_Y_white,(batch_Y_white.shape[0],1))})
            Loss_black = sess.run(loss_pointwise,feed_dict={x:batch_X_black,y:np.reshape(batch_Y_black,(batch_Y_black.shape[0],1))})
            Loss_white = np.reshape(Loss_white,max(Loss_white.shape))
            Loss_black = np.reshape(Loss_black,max(Loss_black.shape))

            white_NoOutlier = np.array(Loss_white<=np.quantile(Loss_white,Q0))
            black_NoOutlier = np.array(Loss_black<=np.quantile(Loss_black,Q1))

            batch_X = np.concatenate((batch_X_white[white_NoOutlier],batch_X_black[black_NoOutlier]))
            batch_Y = np.concatenate((batch_Y_white[white_NoOutlier],batch_Y_black[black_NoOutlier]))

        sess.run(train_step,feed_dict={x:batch_X,y:np.reshape(batch_Y,(batch_Y.shape[0],1))})
        Loss.append(sess.run(loss,feed_dict={x:dataX,y:np.reshape(dataY,(dataY.shape[0],1))}))
        if step_ % 20 == 0:
            time_ = datetime.datetime.now().strftime('%Y-%D %H:%M:%S')
            print(time_ + '\t'+str(step_) +'\t'+ str(Loss[-1]))

    if y_track or beta_track:
        for i in range(100):
            index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]
            batch_X = dataX[index_batch]
            batch_Y = dataY[index_batch]

            if Q1<1 or Q0<1:
                batch_X_white = batch_X[batch_Y==0]
                batch_X_black = batch_X[batch_Y==1]
                batch_Y_white = batch_Y[batch_Y==0]
                batch_Y_black = batch_Y[batch_Y==1]

                Loss_white = sess.run(loss_pointwise,feed_dict={x:batch_X_white,y:np.reshape(batch_Y_white,(batch_Y_white.shape[0],1))})
                Loss_black = sess.run(loss_pointwise,feed_dict={x:batch_X_black,y:np.reshape(batch_Y_black,(batch_Y_black.shape[0],1))})
                Loss_white = np.reshape(Loss_white,max(Loss_white.shape))
                Loss_black = np.reshape(Loss_black,max(Loss_black.shape))

                white_NoOutlier = np.array(Loss_white<=np.quantile(Loss_white,Q0))
                black_NoOutlier = np.array(Loss_black<=np.quantile(Loss_black,Q1))

                batch_X = np.concatenate((batch_X_white[white_NoOutlier],batch_X_black[black_NoOutlier]))
                batch_Y = np.concatenate((batch_Y_white[white_NoOutlier],batch_Y_black[black_NoOutlier]))

            sess.run(train_step,feed_dict={x:batch_X,y:np.reshape(batch_Y,(batch_Y.shape[0],1))})
            Loss.append(sess.run(loss,feed_dict={x:dataX,y:np.reshape(dataY,(dataY.shape[0],1))}))
            beta_HAT.append(sess.run(beta))
            Y_HAT.append(sess.run(y_hat,feed_dict={x:dataX,y:np.reshape(dataY,(dataY.shape[0],1))}))

        beta_HAT = np.array(beta_HAT)
        beta_HAT = np.reshape(beta_HAT,beta_HAT.shape[:2]).T

    if standardize:
        coefs = np.zeros((dataX.shape[1]+1))
        betas = sess.run(beta)
        a = pd.DataFrame(betas)
        a.loc[a.index[np.abs(a.loc[:,0])<=0.05],0] = 0
        betas = a[0]
        for i in range(dataX.shape[1]):
            coefs[i] = stdY/std_[i]*betas[i]
            coefs[-1] -= mean_[i]/std_[i]*betas[i]*stdY
        coefs[-1] += meanY
    else:
        coefs = np.zeros((dataX.shape[1]+1))
        betas = sess.run(beta)
        a = pd.DataFrame(betas)
        a.loc[a.index[np.abs(a.loc[:,0])<=0.05],0] = 0
        coefs[:-1] = a[0]
        coefs[-1] = sess.run(intercept)
    tf.reset_default_graph()
    return(coefs,Loss,Y_HAT,beta_HAT)

def lasso_reg(dataX,dataY,max_iter,batch_size,cv_method = 'auc',standardize=False,start_point='ridge',Q1=1,Q0=0.98):
    lambd = [1e-1,1e-2,1e-3,1e-4,1e-5]
    score = []
    for l in lambd:
        score_ = 0
        for _ in range(2):
            print('CV loop No.'+str(_)+' for lambda='+str(l))
            X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.33, random_state=42)
            coef_ = mini_lasso(X_train,Y_train,l,max_iter=max_iter,batch_size=batch_size,\
                               standardize=standardize,start_point=start_point,Q1=Q1,Q0=Q0)[0]
            Y_hat = np.matmul(np.append(X_test,np.ones((len(X_test),1)),axis=1),coef_)
            if cv_method=='auc':
                try:
                    score_ += roc_auc_score(Y_test,Y_hat)
                except:
                    score_ += 0
            if cv_method=='f1_score':
                Y_hat = pd.DataFrame(Y_hat)
                Y_hat.loc[Y_hat.index[Y_hat.loc[:,0]<0.5],0] = 0
                Y_hat.loc[Y_hat.index[Y_hat.loc[:,0]>=0.5],0] = 1
                Y_hat = np.array(Y_hat[0])
                try:
                    score_ += calculate_f1(Y_test,Y_hat)
                except:
                    score_ += 0
        score.append(score_/3)
    lambd_ = lambd[[i for i in range(len(score)) if score[i] == max(score)][0]]
    print('Set lambda='+str(lambd_)+' and fit model...')
    coefs,Loss,Y_HAT,beta_HAT = mini_lasso(dataX,dataY,lambd_,max_iter,y_track=True,\
                                           standardize=standardize, start_point=start_point,Q1=Q1,Q0=Q0)
    print("Lasso picked " + str(sum(coefs != 0)-1) + \
          " variables and eliminated the other " +  str(sum(coefs == 0)) + " variables")
    return(coefs,sum(coefs != 0)-1,lambd_,Loss,Y_HAT,beta_HAT)

def mini_Glasso(dataX_,dataY_,K,lambd,max_iter,batch_size=1024,standardize=False):
    '''
    This function requires feature matrix, a column of label from training data, a dictionary K of 
    number of features in each group, hyper varaible lambd, and maxinum number of iteration max_iter.
    The function give out the fitted coefficients of group lasso as well as trace of loss function.
    The group lasso would standardize the input data automatically!
    The corresponding positive defnite matrix k if defined by K, where k is a diagnostic matrix with 
    elements as number of features in each group.
    
    The optimazation would stop if loss function reduces less than 0.01%, or number of iterations reaches
    max_iter.
    '''
    dataX = copy.deepcopy(dataX_)
    dataY = copy.deepcopy(dataY_)
    if standardize:
        std_ = []
        mean_ = []
        meanY = np.mean(dataY)
        stdY = np.std(dataY)
        dataY = (dataY-meanY)/stdY
        for i in range(dataX.shape[1]):
            std_.append(np.std(dataX[:,i]))
            mean_.append(np.mean(dataX[:,i]))
            dataX[:,i] = (dataX[:,i]-mean_[i])/std_[i]
    tf.reset_default_graph()
    x = tf.placeholder(dtype = tf.float32, shape=[None,dataX.shape[1]])
    y = tf.placeholder(dtype = tf.float32, shape=[None,1])
    l = tf.constant(lambd,dtype=tf.float32)
    beta = tf.Variable(tf.zeros([dataX.shape[1],1]),name = 'beta',dtype=tf.float32)
    if standardize:
        y_hat_linear = tf.matmul(x,beta)
    else:
        intercept = tf.Variable(tf.zeros([1,1]),name = 'intercept',dtype=tf.float32)
        y_hat_linear = tf.matmul(x,beta) + intercept
    y_hat = tf.sigmoid(y_hat_linear)
    loss = 10*tf.reduce_mean(-y*tf.log(tf.clip_by_value(y_hat,1e-10,1.0))-(1-y)*tf.log(tf.clip_by_value(1-y_hat,1e-10,1.0)))
    index_cum = 0
    for group in K:
        index_cum_ = index_cum
        index_cum += K[group]
        mat_group = tf.constant(np.diag([K[group]]*K[group]),dtype=tf.float32)
        beta_group = tf.slice(beta,[index_cum_,0],[K[group],1])
        loss = loss + l*tf.sqrt(tf.matmul(tf.matmul(tf.transpose(beta_group),mat_group),beta_group)+1e-10)
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.5,global_step,decay_steps=max_iter/10,decay_rate=0.95)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init_op)
    Loss = []
    index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]
    batch_X = dataX[index_batch]
    batch_Y = dataY[index_batch]
    sess.run(train_step,feed_dict={x:batch_X,y:np.reshape(batch_Y,(batch_Y.shape[0],1))})
    Loss.append(sess.run(loss,feed_dict={x:dataX,y:np.reshape(dataY,(dataY.shape[0],1))}))
    sess.run(train_step,feed_dict={x:batch_X,y:np.reshape(batch_Y,(batch_Y.shape[0],1))})
    Loss.append(sess.run(loss,feed_dict={x:dataX,y:np.reshape(dataY,(dataY.shape[0],1))}))
    step_ = 2
    while (((Loss[-2]-Loss[-1])/np.abs(Loss[-2])>.0001) or ((Loss[-2]-Loss[-1])<0) or step_<30) & (step_<=max_iter):
        step_ += 1
        index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]
        batch_X = dataX[index_batch]
        batch_Y = dataY[index_batch]
        sess.run(train_step,feed_dict={x:batch_X,y:np.reshape(batch_Y,(batch_Y.shape[0],1))})
        Loss.append(sess.run(loss,feed_dict={x:dataX,y:np.reshape(dataY,(dataY.shape[0],1))}))
        if step_ % 10 == 0:
            time_ = datetime.datetime.now().strftime('%Y-%D %H:%M:%S')
            print(time_ + '\t'+str(step_) +'\t'+ str(Loss[-1]))
    if standardize:
        coefs = np.zeros((dataX.shape[1]+1))
        betas = sess.run(beta)
        a = pd.DataFrame(betas)
        a.loc[a.index[np.abs(a.loc[:,0])<=0.01],0] = 0
        betas = a[0]
        for i in range(dataX.shape[1]):
            coefs[i] = stdY/std_[i]*betas[i]
            coefs[-1] -= mean_[i]/std_[i]*betas[i]*stdY
        coefs[-1] += meanY
    else:
        coefs = np.zeros((dataX.shape[1]+1))
        betas = sess.run(beta)
        a = pd.DataFrame(betas)
        a.loc[a.index[np.abs(a.loc[:,0])<=0.01],0] = 0
        coefs[:-1] = a[0]
        coefs[-1] = sess.run(intercept)
    tf.reset_default_graph()
    return(coefs,Loss)
    
def Glasso_reg(dataX,dataY,K,max_iter,batch_size,cv_method = 'auc',standardize=False):
    lambd = [1, 0.5, 0.1, 0.05, 0.001,0.0001]
    score = []
    for l in lambd:
        score_ = 0
        for _ in range(3):
            X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.33, random_state=42)
            coef_ = mini_Glasso(X_train,Y_train,K,l,max_iter=max_iter,batch_size=batch_size,standardize=standardize)[0]
            Y_hat = np.matmul(np.append(X_test,np.ones((len(X_test),1)),axis=1),coef_)
            if cv_method=='auc':
                try:
                    score_ += roc_auc_score(Y_test,Y_hat)
                except:
                    score_ += 0
            if cv_method=='f1_score':
                Y_hat = pd.DataFrame(Y_hat)
                Y_hat.loc[Y_hat.index[Y_hat.loc[:,0]<0.5],0] = 0
                Y_hat.loc[Y_hat.index[Y_hat.loc[:,0]>=0.5],0] = 1
                Y_hat = np.array(Y_hat[0])
                try:
                    score_ += calculate_f1(Y_test,Y_hat)
                except:
                    score_ += 0
        score.append(score_/3)
    lambd_ = lambd[[i for i in range(len(score)) if score[i] == max(score)][0]]
    coefs,Loss = mini_Glasso(dataX,dataY,K,lambd_,max_iter)
    print("Group Lasso picked " + str(sum(coefs != 0)-1) + \
          " variables and eliminated the other " +  str(sum(coefs == 0)) + " variables")
    return(coefs,sum(coefs != 0)-1,lambd_,Loss)






if __name__ == '__main__':
    ###样例数据
    df = pd.read_csv('test_sample_data.csv',sep='\t')
    label_col = 'is_black'
    var_cols = list(df.columns[1:-1])
    ###分省份画图示例
    import random
    province = ['河南', '北京', '河北', '辽宁', '江西', '上海', '安徽', '江苏', \
                '湖南', '浙江', '海南', '广东', '湖北', '黑龙江', '澳门', '陕西', \
                '四川', '内蒙古', '重庆', '云南', '贵州', '吉林', '山西', '山东', \
                '福建', '青海', '天津']
    df['province'] = random.choices(province,k=len(df))
    df['industry'] = random.choices(['红警新兴金融类','红警投资公司','红警农民专业合作社','红警融资租赁'],k=len(df))
    ###数据准备
    df2 = df.fillna(0)
    valid_index = np.sum(df2.iloc[:,1:-3]>0,axis=0)/len(df2)>0.004
    valid_index = [i for i in range(1,len(valid_index)+1) if valid_index[i-1]]
    valid_index.append(-1)
    valid_index.append(-3)
    df2 = df2.iloc[:,valid_index]
    X_cols = df2.columns.tolist()[:-2]
    ###变量二值化示例
    df2,new_X_cols = MultilevelVariable(df2,'industry',X_cols)
    ###########建模示例########
    from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV
    from sklearn.metrics import roc_auc_score
    model = LassoLarsCV() 
    model.fit(X_train, y_train)
    ###变量分行业汇总示例
    X_ = MultilevelCoef(model.coef_,model.intercept_,new_X_cols,level_names=list(set(df['industry'])),indicator_col='industry')
    model = LassoLarsCV() 
    X_,AUC,_,__ = byIndustryModel(df.fillna(0),X_cols,'is_black','industry',0.2,model)

    model = LassoClassifier()
    X_,AUC,_,__ = byIndustryModel(df.fillna(0),X_cols,'is_black','industry',0.2,model)

