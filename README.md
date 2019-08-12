*LinearModel.py* : 

    模块说明：
        1. 由于sklearn自带的Logistics回归模型在l1惩罚项下拟合较慢，本模块针对几个包含l1罚项的线性回归二分类模型编写了
           tensorflow的实现，增加了拟合速度其调用方式基本与sklearn中的模型完全相同，即支持fit()、predict()方法，另包
           含plotLoss()绘制损失函数曲线方法，包含coef_、intercept_等属性。
        2. 针对需要划分行业的lasso建模问题，设计了将行业indicator加入所有的回归系数中，即加入行业与所有变量的交互效应，再用lasso自动筛选
           筛选过后将得到的回归系数还原到各个行业下。这样的好处是建立统一的回归模型，不用分行业建立单独的模型，节约了自由度，利用了更多的数据信息。
        3. KS曲线绘制函数PlotKS
        4. 本模块对Logistic的主要改动如下：
            1）模型框架上，为适应WOE等单变量变换算法，设计了正系数逻辑回归。所有系数均为正值。
            2）最优化路径上，替换为随机梯度下降；迭代起始点选取为岭回归估计量。
            3）异常值处理上，每个batch排除损失函数最大的(1-Q)*100%的数据，余下数据参与调参。
            4）参数估计上，采用模型收敛之后100次迭代过程中每一次迭代所得参数的均值。结合随机梯度下降，相当于做了bagging。
    版本信息：
        sklearn 0.20.1
        tensorflow 1.13.1
        matplotlib 3.0.2
        python 3.7.1

*MonoLogitTrans.py* : 

    模块说明：
        本模块提供了类似于WOE编码的一个单变量连续函数编码。该编码具有以下几个性质：
            1) 在二分类问题中，和目标变量取值为1的概率正相关
            2) 可以通过调整参数，使得该编码为单调编码，即为变量原始值的单调变换
            3) 在二分类问题中，该编码和WOE一样，均等价于目标变量取值为1的概率的logit变换
        
        具体方式：
            方式一(method='wide'):
                用多隐层神经网络，拟合单变量X对目标变量Y的Logistic回归。
                在损失函数中，分为交叉熵损失和单调性损失两个部分。
                其中，f(x)为该单变量转化函数，则单调性损失定义如下：
                    $\text{loss_monotonous}(x,f(x)) = 1 - |(\rho(x,f(x)))|$
                    $\rho(x,y)$为皮尔森线性相关系数
                最终损失函数为
                    loss = cross_entropy($\sigma$(f(x)),y)+lambda_monotonous*loss_monotonous(x,f(x))
            方式二(method='strict'):
                每一个隐层内部权值的符号均相同，保证神经网络拟合出一个单调函数f(x)
                损失函数为交叉熵损失
                    loss = cross_entropy($\sigma$(f(x)),y)
    版本信息：
        sklearn 0.20.1
        tensorflow 1.13.1
        python 3.7.1
        