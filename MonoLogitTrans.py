'''
Author: 苏冠旭
update date : 20190716

本模块提供了一个单变量编码，用于在拟合线性模型之前进行数据转化，提升模型效果。

该编码具有以下几个性质：
    1) 在二分类问题中，和目标变量取值为1的概率正相关
    2) 可以通过调整参数，使得该编码为单调编码，即为变量原始值的单调变换
    3) 在二分类问题中，该编码和WOE一样，均等价于目标变量取值为1的概率的logit变换

具体方式：
    方式一(method='wide'):
        用多隐层神经网络，拟合单变量X对目标变量Y的Logistic回归。
        在损失函数中，分为交叉熵损失和单调性损失两个部分。
        其中，f(x)为该单变量转化函数，则单调性损失定义如下：
            loss_monotonous(x,f(x)) = 1 - tf.abs(rho(x,f(x)))
            rho(x,y)为皮尔森线性相关系数
        最终损失函数为
            loss = cross_entropy(sigmoid(f(x)),y)+lambda_monotonous*loss_monotonous(x,f(x))
    方式二(method='strict'):
        每一个隐层内部权值的符号均相同，保证神经网络拟合出一个单调函数f(x)
        损失函数为交叉熵损失
            loss = cross_entropy(sigmoid(f(x)),y)

特殊情况:
    1) 如果x只有一种或两种取值，则若rho(x,y)>0，返回x，rho(x,y)<0返回-x
    2) 如果x和y之间关系极不明显，则f(x)趋近于常数，记s(x)为x标准化之后的值，若rho(x,y)>0，返回s(x)，rho(x,y)<0返回-s(x)

Todo:
    1) 对于方式一，放宽单调性损失的限制，比如改为秩相关系数等。但是求秩函数不可微，所以还需要考虑其他办法。
    2) 对于方式一，寻找合理调参方式，比如CV等


-------------------------------------------------------------------------------------------
-------------------------------------2019-07-18升级说明-------------------------------------
-------------------------------------------------------------------------------------------

增加功能：
    1. 增加method属性，用于选择转化函数是否是严格单调函数
    2. 增加target属性，用于选择目标变量是连续变量还是二分类变量
    3. 完善transform属性对异常值的处理

'''


import tensorflow as tf
import numpy as np
import copy
import pandas as pd
import random
import datetime
import pickle
import matplotlib.pyplot as plt
from sklearn import datasets
import random
import warnings

class MonoLogitTrans(object):
    '''
Continuous monotone function of Single variable encoding 
based on logistic regression. The encoded value is the 
logit of probability of y=1 under monotonous contrains.

Parameters
----------
max_iter : int, default: 100
    Maximum number of iterations of the optimization algorithm.

batch_size : int, default: 1024
    Number of rows of each batch of the batch training.
    If equal to number of rows of training data, then the algorithm
    would be equivalent to full scale training.

num_hidden : int, default: 10
    Number of nodes in hidden layers.

lambda_monotonous : float, default: 1
    weight of monotone loss in loss function
    If set to be 0, the transformation is equivalent to fitting
    a logistic regression on the single varaible.

method : str, can be choosed from 'wide' and 'strict'
    If 'wide' is choosed, then monotonicity is reached by adjusting
    loss function, and may not be strictly satisfied.
    If 'strict' is choosed, then the signs of weights inside the same
    hidden layer are limited to be the same. The fitted transformation
    function would be a monotone function for sure.

target : str, can be choosed from 'classifier' and 'regression'
    'regression' should be choosed if Y is continuous.
    'classifier' should be choosed if Y is binary varaible and has values of 0 or 1.

Attributes
----------
weights: list of length 3
    weights in the encoding network

biases: list of length 3
    biases in the encoding network 

Methods
-------
fit: optimize the encoding network

transform: perform the tranformation given new varaible values.

save_parameter: save the parameters into a pickle file

load_parameter: load the parameters from a pickle file

    '''

    def __init__(self,max_iter=150,batch_size=1024,num_hidden=10,lambda_monotonous=1,method='wide',target='classifier'):
        if method not in ['wide','strict']:
            raise ValueError('method could only be "wide" or "strict"')
        if target not in ['classifier','regression']:
            raise ValueError('target could only be "classifier" or "regression"')
        if method=='strict' and num_hidden<10:
            print('Greater value is suggested for "num_hidden" when strict monotone constraint is choosed!') 
        self.max_iter=max_iter
        self.target=target
        self.batch_size=batch_size
        self.num_hidden=num_hidden
        self.lambda_monotonous=lambda_monotonous
        self.method = method
        self.relation=None
        self.x_mean_data = None
        self.x_std_data = None

    def fit(self,dataX,dataY):
        self.relation = np.corrcoef(dataX,dataY)[0,1]>0
        if len(set(dataX))<3:
            if self.relation:
                self.weights = 1
                self.biases = 0
                self.x_mean,self.x_std = None,1e-10
                self.x_max_data = np.max(dataX)
                self.x_min_data = np.min(dataX)

            else:
                self.weights = -1
                self.biases = 0
                self.x_mean,self.x_std = None,1e-10
                self.x_max_data = np.max(dataX)
                self.x_min_data = np.min(dataX)
        else:
            if self.method == 'wide':
                w1_,w2_,w3_,b1_,b2_,b3_,step_,Loss,x_mean,x_std = fit_network(
                    dataX,dataY,
                    num_hidden=self.num_hidden,
                    lambda_monotonous=self.lambda_monotonous,
                    max_iter=self.max_iter,
                    batch_size=self.batch_size,
                    target=self.target
                    )
            elif self.method == 'strict':
                w1_,w2_,w3_,b1_,b2_,b3_,step_,Loss,x_mean,x_std = fit_network_strict_mono(
                    dataX,dataY,
                    num_hidden=self.num_hidden,
                    max_iter=self.max_iter,
                    batch_size=self.batch_size,
                    target=self.target
                    )
            self.weights = [w1_,w2_,w3_]
            self.biases = [b1_,b2_,b3_]
            self.Loss = Loss
            self.x_mean,self.x_std = x_mean,x_std+1e-10
            self.x_mean_data = np.mean(dataX)
            self.x_std_data = np.std(dataX)+1e-10
            self.x_max_data = np.max(dataX)
            self.x_min_data = np.min(dataX)


            

    def set_weights(self,w1_,w2_,w3_):
        self.weights = [w1_,w2_,w3_]

    def set_biases(self,b1_,b2_,b3_):
        self.biases = [b1_,b2_,b3_] 

    def transform(self,dataX):

        if type(self.weights) is int:
            if self.weights==1:
                return(dataX)
            elif self.weights==-1:
                return(-dataX)
        else:
            a = pd.DataFrame(dataX)
            a.loc[a.index[a.loc[:,0]>self.x_max_data],0] = self.x_max_data
            a.loc[a.index[a.loc[:,0]<self.x_min_data],0] = self.x_min_data
            dataX = a[0].values
            X_trans = transX(
                    dataX,
                    self.weights[0],self.weights[1],self.weights[2],
                    self.biases[0],self.biases[1],self.biases[2]
                )
            X_trans = (X_trans-self.x_mean)/self.x_std
            X_trans_origin = (dataX-self.x_mean_data)/self.x_std_data
            if self.x_std>1e-1:
                a = pd.DataFrame(X_trans)
                a.loc[a.index[a.loc[:,0]>10],0] = 10
                a.loc[a.index[a.loc[:,0]<-10],0] = -10
                X_trans = a[0].values
                return(X_trans)
            else:
                if self.relation:
                    return(X_trans_origin)
                else:
                    return(-X_trans_origin)

    def save_parameter(self,path):
        with open(path, 'wb') as f:
            pickle.dump([self.weights,self.biases,self.x_mean,self.x_std,self.relation,self.x_mean_data,self.x_std_data,self.x_max_data,self.x_min_data], f)

    def load_parameter(self,path):
        with open(path,'rb') as f:
            self.weights,self.biases,self.x_mean,self.x_std,self.relation,self.x_mean_data,self.x_std_data,self.x_max_data,self.x_min_data = pickle.load(f,encoding='latinl')

    def plot(self):
        x_origin_mini = (np.array(range(2000))/2000)*(self.x_max_data-self.x_min_data)+self.x_min_data
        x_trans_mini = self.transform(x_origin_mini)
        plt.plot(x_origin_mini,x_trans_mini,'o-')

def transX(dataX_,w1,w2,w3,b1,b2,b3):
    x = np.reshape(dataX_,(dataX_.shape[0],1))
    h1 = np.matmul(x,w1) + b1
    #h1 = 1/(1+np.exp(-h1))
    h1 = np.tanh(h1)
    #h1[h1<0]=0
    #h1[h1>6]=6
    h2 =np.matmul(h1,w2) + b2
    #h2 = 1/(1+np.exp(-h2))
    h2 = np.tanh(h2)
    #h2[h2<0]=0
    #h2[h2>6]=6
    x_trans = np.matmul(h2,w3)+b3
    #x_trans = h2
    #x_trans = 1/(1+np.exp(-x_trans))
    x_trans = np.reshape(x_trans,dataX_.shape)
    #x_trans = (x_trans-np.mean(x_trans))/np.std(x_trans)
    return(x_trans)


def make_positive(input_tensor):
    #out_ = input_tensor*tf.cast(input_tensor>0, tf.float32)
    out_ = tf.clip_by_value(input_tensor,0.0,9999)
    return out_

def make_greater_than_1(input_tensor):
    #out_ = input_tensor*tf.cast(input_tensor>0, tf.float32)
    out_ = tf.clip_by_value(input_tensor,1e-10,9999)
    return out_

def become_sign(input_tensor):
    #out_ = input_tensor*tf.cast(input_tensor>0, tf.float32)
    out_ = tf.sign(input_tensor)
    return out_


def fit_network(dataX_,dataY_,num_hidden=10,lambda_monotonous=2,max_iter=150,batch_size=1024,target='classifier'):
    '''
    The optimazation would stop if loss function reduces less than 0.01%, or number of iterations reaches
    max_iter.
    '''
    tf.reset_default_graph()
    dataX = copy.deepcopy(dataX_)
    dataX = np.reshape(dataX,(dataX.shape[0],1))
    dataY = copy.deepcopy(dataY_)
    dataY = np.reshape(dataY,(dataY.shape[0],1))
    dataX_stand = (dataX-np.mean(dataX))/np.std(dataX)
    dataX_stand = np.reshape(dataX_stand,(dataX_stand.shape[0],1))
    tf.reset_default_graph()
    x = tf.placeholder(dtype = tf.float32, shape=[None,1],name='x')
    x_stand = tf.placeholder(dtype = tf.float32, shape=[None,1],name='x_stand')
    y = tf.placeholder(dtype = tf.float32, shape=[None,1],name='y')
    regularizer = tf.contrib.layers.l1_regularizer(1.0)
    ###主网络
    with tf.variable_scope('weight', initializer=tf.random_normal_initializer(),regularizer=regularizer):
        w1 = tf.Variable(tf.random_normal([1,num_hidden], stddev=0.1,mean=0),name = 'w1',dtype=tf.float32)
        w2 = tf.Variable(tf.random_normal([num_hidden,num_hidden], stddev=0.1,mean=0),name = 'w2',dtype=tf.float32)
        w3 = tf.Variable(tf.random_normal([num_hidden,1], stddev=0.1,mean=0),name = 'w3',dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([num_hidden]),name = 'b1',dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([num_hidden]),name = 'b2',dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([1]),name = 'b3',dtype=tf.float32)
    h1 = tf.nn.tanh(tf.matmul(x,w1) + b1)
    h2 = tf.nn.tanh(tf.matmul(h1,w2) + b2)
    #h2 = tf.matmul(h1,w2) + b2
    x_trans_linear = tf.matmul(h2,w3)+b3 #linear predictor
    #x_trans_linear = h2
    x_trans = tf.sigmoid(x_trans_linear) #predicted probability
    if target=='classifier':
        loss_y = tf.reduce_mean(-y*tf.log(tf.clip_by_value(x_trans,1e-10,1.0-1e-10))-(1-y)*tf.log(tf.clip_by_value(1-x_trans,1e-10,1.0-1e-10)))
    elif target=='regression':
        loss_y = tf.losses.mean_squared_error(y,x_trans_linear)
    ###单调限制网络
    w1_1 = tf.Variable(tf.random_normal([1], stddev=0.1,mean=1),name = 'w1_1',dtype=tf.float32,constraint=make_positive)
    w1_3 = tf.Variable(tf.random_normal([1], stddev=0.1,mean=1),name = 'w1_3',dtype=tf.float32,constraint=make_positive)
    w1_5 = tf.Variable(tf.random_normal([1], stddev=0.1,mean=1),name = 'w1_5',dtype=tf.float32,constraint=make_positive)
    w1_exp = tf.Variable(tf.random_normal([1], stddev=0.1,mean=1),name = 'w1_exp',dtype=tf.float32,constraint=make_positive)
    #w_sign = tf.Variable(tf.random_normal([1], stddev=0.1,mean=1),name = 'w_sign',dtype=tf.float32,constraint=tf.sign)
    b1_1 = tf.Variable(tf.zeros([1]),name = 'b1_1',dtype=tf.float32)
    #x_mono_trans = w_sign*(x*w1_1+(x**3)*w1_3+(x**5)*w1_5+np.exp(1)**(x)*w1_exp)+ b1_1 #monotonou transformation
    #x_mono_trans = w_sign*x
    x_mono_trans = x-tf.reduce_mean(x)
    x_trans_linear_centre = x_trans_linear-tf.reduce_mean(x_trans_linear)
    loss_monotonous = 1-tf.abs(x_mono_trans*x_trans_linear_centre/(tf.sqrt(tf.reduce_sum(x_mono_trans*x_mono_trans))*\
        tf.sqrt(tf.reduce_sum(x_trans_linear_centre*x_trans_linear_centre)))) #cosine similirity/Pearson correlation


    loss = 10*loss_y+lambda_monotonous*loss_monotonous # the final loss
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.5,global_step,decay_steps=max_iter/3,decay_rate=0.95)
    train_step_y = tf.train.AdamOptimizer(learning_rate).minimize(loss_y,global_step=global_step)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    train_step_mono = tf.train.AdamOptimizer(learning_rate).minimize(loss_monotonous,global_step=global_step)
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init_op)
    Loss = []
    for i in range(10):
        index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]
        batch_X = dataX[index_batch]
        batch_Y = dataY[index_batch]
        batch_X_stand = dataX_stand[index_batch]
        for i in range(3):
            sess.run(train_step_y,feed_dict={x:batch_X,y:batch_Y,x_stand:batch_X_stand})
            Loss.append(sess.run(loss_y,feed_dict={x:dataX,y:dataY}))

    step_ = 2
    while (((Loss[-2]-Loss[-1])/(np.abs(Loss[-2])+1e-10)>.001) or ((Loss[-2]-Loss[-1])<0) or step_<30) & (step_<=max_iter):
        step_ += 1
        index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]
        batch_X = dataX[index_batch]
        batch_Y = dataY[index_batch]
        batch_X_stand = dataX_stand[index_batch]
        sess.run(train_step,feed_dict={x:batch_X,y:batch_Y,x_stand:batch_X_stand})
        Loss.append(sess.run(loss_y,feed_dict={x:dataX,y:dataY}))
        if step_ % 1000 == 0:
            time_ = datetime.datetime.now().strftime('%Y-%D %H:%M:%S')
            print(time_ + '\t'+str(step_) +'\t'+ str(Loss[-1]))

    w1_ = sess.run(w1)
    w2_ = sess.run(w2)
    w3_ = sess.run(w3)
    b1_ = sess.run(b1)
    b2_ = sess.run(b2)
    b3_ = sess.run(b3)
    X_trans_L = sess.run(x_trans_linear,feed_dict={x:dataX})
    x_mean = np.mean(X_trans_L)
    x_std = np.std(X_trans_L)
    tf.reset_default_graph()
    return(w1_,w2_,w3_,b1_,b2_,b3_,step_,Loss,x_mean,x_std)



def fit_network_strict_mono(dataX_,dataY_,num_hidden=10,lambda_monotonous=2,max_iter=150,batch_size=1024,target='classifier'):
    '''
    The optimazation would stop if loss function reduces less than 0.01%, or number of iterations reaches
    max_iter.
    '''
    tf.reset_default_graph()
    dataX = copy.deepcopy(dataX_)
    dataX = np.reshape(dataX,(dataX.shape[0],1))
    dataY = copy.deepcopy(dataY_)
    dataY = np.reshape(dataY,(dataY.shape[0],1))
    dataX_stand = (dataX-np.mean(dataX))/np.std(dataX)
    dataX_stand = np.reshape(dataX_stand,(dataX_stand.shape[0],1))
    tf.reset_default_graph()
    x = tf.placeholder(dtype = tf.float32, shape=[None,1],name='x')
    x_stand = tf.placeholder(dtype = tf.float32, shape=[None,1],name='x_stand')
    y = tf.placeholder(dtype = tf.float32, shape=[None,1],name='y')
    regularizer = tf.contrib.layers.l1_regularizer(0.1)
    ###主网络
    with tf.variable_scope('weight', initializer=tf.random_normal_initializer(),regularizer=regularizer):
        w1_pos = tf.Variable(tf.random_normal([1,num_hidden], stddev=0.1,mean=0.5),name = 'w1_pos',dtype=tf.float32,constraint=make_greater_than_1)
        w2_pos = tf.Variable(tf.random_normal([num_hidden,num_hidden], stddev=0.1,mean=0.5),name = 'w2_pos',dtype=tf.float32,constraint=make_positive)
        w3_pos = tf.Variable(tf.random_normal([num_hidden,1], stddev=0.1,mean=0.5),name = 'w3_pos',dtype=tf.float32,constraint=make_greater_than_1)
        w1_sign = tf.Variable(tf.random_normal([1], stddev=0.1,mean=0),name = 'w1_sign',dtype=tf.float32)
        w2_sign = tf.Variable(tf.random_normal([1], stddev=0.1,mean=0),name = 'w2_sign',dtype=tf.float32)
        w3_sign = tf.Variable(tf.random_normal([1], stddev=0.1,mean=0),name = 'w3_sign',dtype=tf.float32)
        w1 = tf.multiply(w1_pos,w1_sign,name='w1')
        w2 = tf.multiply(w2_pos,w2_sign,name='w2')
        w3 = tf.multiply(w3_pos,w3_sign,name='w3')
    b1 = tf.Variable(tf.zeros([num_hidden]),name = 'b1',dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([num_hidden]),name = 'b2',dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([1]),name = 'b3',dtype=tf.float32)
    h1 = tf.nn.tanh(tf.matmul(x,w1) + b1) + 1e-1*tf.random_normal([num_hidden])
    h2 = tf.nn.tanh(tf.matmul(h1,w2) + b2) + 1e-1*tf.random_normal([num_hidden])
    x_trans_linear = tf.matmul(h2,w3)+b3 #linear predictor
    x_trans = tf.sigmoid(x_trans_linear) #predicted probability

    if target=='classifier':
        loss = tf.reduce_mean(-y*tf.log(tf.clip_by_value(x_trans,1e-10,1.0-1e-10))-(1-y)*tf.log(tf.clip_by_value(1-x_trans,1e-10,1.0-1e-10)))
    elif target=='regression':
        loss = tf.losses.mean_squared_error(y,x_trans_linear)

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.5,global_step,decay_steps=max_iter/3,decay_rate=0.95)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init_op)
    Loss = []
    for i in range(10):
        index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]
        batch_X = dataX[index_batch]
        batch_Y = dataY[index_batch]
        batch_X_stand = dataX_stand[index_batch]
        for i in range(3):
            sess.run(train_step,feed_dict={x:batch_X,y:batch_Y,x_stand:batch_X_stand})
            Loss.append(sess.run(loss,feed_dict={x:dataX,y:dataY}))

    step_ = 2
    while (((Loss[-2]-Loss[-1])/(np.abs(Loss[-2])+1e-10)>.001) or ((Loss[-2]-Loss[-1])<0) or step_<30) & (step_<=max_iter):
        step_ += 1
        index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]
        batch_X = dataX[index_batch]
        batch_Y = dataY[index_batch]
        batch_X_stand = dataX_stand[index_batch]
        sess.run(train_step,feed_dict={x:batch_X,y:batch_Y,x_stand:batch_X_stand})
        Loss.append(sess.run(loss,feed_dict={x:dataX,y:dataY}))
        if step_ % 1000 == 0:
            time_ = datetime.datetime.now().strftime('%Y-%D %H:%M:%S')
            print(time_ + '\t'+str(step_) +'\t'+ str(Loss[-1]))

    w1_ = sess.run(w1)
    w2_ = sess.run(w2)
    w3_ = sess.run(w3)
    b1_ = sess.run(b1)
    b2_ = sess.run(b2)
    b3_ = sess.run(b3)
    X_trans_L = sess.run(x_trans_linear,feed_dict={x:dataX})
    x_mean = np.mean(X_trans_L)
    x_std = np.std(X_trans_L)
    tf.reset_default_graph()
    return(w1_,w2_,w3_,b1_,b2_,b3_,step_,Loss,x_mean,x_std)


def PlotComparableHistogram(variable_,lable_,log=True):
    '''
        Author: 苏冠旭
        绘制对比直方图
        Input:
            variable_是需要绘制的直方图的值
            lable_是和variable_长度相同的标签
            save_dir是存储路径
            file_name是图片存储的文件名，不带路径。如果加后缀名，则图片自动以该后缀名格式存储。若不指定，则按照variable_的name进行命名
            title是图片的标题，若不指定，则按照variable_的name生成title
            log判断是否做对数变换
        Output:
            None
    '''
    ###数据重新编码index，避免index不连续而出错
    variable_.index = range(len(variable_))
    lable_.index = range(len(lable_))
    ###色彩条
    color_num = 0
    color_bar = ['darkblue','darkred','darkgoldenrod','yellowgreen']
    ###对lable_中的每一个unique值进行循环，作图
    lable_set = list(set(lable_))
    fig,ax=plt.subplots(1,1)
    for each_lable in lable_set:
        index_tmp = [i for i in range(len(lable_)) if lable_[i]==each_lable]
        ax.hist(variable_[index_tmp], density =1,facecolor=color_bar[color_num],range=(min(variable_),max(variable_)),\
                 label = each_lable,edgecolor=color_bar[color_num],log=log,rwidth=1,alpha=0.5)
        color_num += 1
    ax.legend(lable_set)
    plt.show()


def PlotKS(preds, labels, n=100, asc=0):
    # preds is score: asc=1
    # preds is prob: asc=0
    pred = preds # 预测值
    bad = labels # 取1为bad, 0为good
    ksds = pd.DataFrame({'bad': bad, 'pred': pred})
    ksds['good'] = 1 - ksds.bad
    
    if asc == 1:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, True])
    elif asc == 0:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, True])
    ksds1.index = range(len(ksds1.pred))
    ksds1['cumsum_good1'] = 1.0*ksds1.good.cumsum()/sum(ksds1.good)
    ksds1['cumsum_bad1'] = 1.0*ksds1.bad.cumsum()/sum(ksds1.bad)
    
    if asc == 1:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, False])
    elif asc == 0:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, False])
    ksds2.index = range(len(ksds2.pred))
    ksds2['cumsum_good2'] = 1.0*ksds2.good.cumsum()/sum(ksds2.good)
    ksds2['cumsum_bad2'] = 1.0*ksds2.bad.cumsum()/sum(ksds2.bad)
    
    # ksds1 ksds2 -> average
    ksds = ksds1[['cumsum_good1', 'cumsum_bad1']]
    ksds['cumsum_good2'] = ksds2['cumsum_good2']
    ksds['cumsum_bad2'] = ksds2['cumsum_bad2']
    ksds['cumsum_good'] = (ksds['cumsum_good1'] + ksds['cumsum_good2'])/2
    ksds['cumsum_bad'] = (ksds['cumsum_bad1'] + ksds['cumsum_bad2'])/2
    
    # ks
    ksds['ks'] = ksds['cumsum_bad'] - ksds['cumsum_good']
    ksds['tile0'] = range(1, len(ksds.ks) + 1)
    ksds['tile'] = 1.0*ksds['tile0']/len(ksds['tile0'])
    
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
    print ('ks_value is ' + str(np.round(ks_value, 4)) + ' at pop = ' + str(np.round(ks_pop, 4)))
    
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
    return ksds

def get_data():
    random.seed(0)
    breastcancer = datasets.load_breast_cancer()
    X = breastcancer['data']
    Y = breastcancer['target']
    def nonliear_trans(x):
        x = (x-np.mean(x))/np.std(x)
        x = 2*x-x**2
        return(x)
    index1 = 0
    index2 = 1
    index3 = 2
    X[:,index1] = nonliear_trans(X[:,index1])
    X[:,index2] = nonliear_trans(X[:,index2])
    X[:,index3] = nonliear_trans(X[:,index3])
    test_index = random.choices(range(len(Y)),k=100)
    train_index = list(set(range(len(Y)))-set(test_index))
    X_train = X[train_index]
    Y_train = Y[train_index]
    X_test = X[test_index]
    Y_test = Y[test_index]
    x1,x2,x3,x1_test,x2_test,x3_test,y,y_test = X_train[:,index1],X_train[:,index2],X_train[:,index3],\
        X_test[:,index1],X_test[:,index2],X_test[:,index3],Y_train,Y_test
    return(x1,x2,x3,x1_test,x2_test,x3_test,y,y_test)


if __name__ == '__main__':

    import random
    random.seed(1)
    X = np.random.uniform(-10,10,size = [10000,20])
    X[:,3] = np.random.choice(2,10000)
    x1 = X[:8000,1]
    x2 = X[:8000,2]
    x3 = X[:8000,3]
    x1_test = X[8000:,1]
    x2_test = X[8000:,2]
    x3_test = X[8000:,3]

    def linear_(x1,x2,x3):
        x2_ = x2
        x2_[x2_>np.quantile(x2_,0.7)] = np.random.uniform(min(x2_),np.quantile(x2_,0.5),size = [sum(x2_>np.quantile(x2_,0.7))])
        return(((np.exp(1)**x1/20-20*x1**2-0.1*x1**3)/20-np.exp(x2_)/10+x2_**2+x3*2+np.random.normal(-10,10,size=[len(x1)])))
    linear_true =linear_(x1,x2,x3)
    y = 1/(1+np.exp(linear_true))
    linear_test =linear_(x1_test,x2_test,x3_test)
    y_test = 1/(1+np.exp(linear_test))
    _ = 0.05
    y[y<_] = 0
    y[y>=_] = 1
    y_test[y_test<_] = 0
    y_test[y_test>=_] = 1


    MonoLogitTrans1 = MonoLogitTrans(max_iter=1000)
    MonoLogitTrans1.fit(x1,y)
    x1_trans = MonoLogitTrans1.transform(x1_test)
    MonoLogitTrans1.save_parameter('x1.txt')

    MonoLogitTrans1_2 = MonoLogitTrans()
    MonoLogitTrans1_2.load_parameter('x1.txt')
    x1_trans_2 = MonoLogitTrans1_2.transform(x1_test)

