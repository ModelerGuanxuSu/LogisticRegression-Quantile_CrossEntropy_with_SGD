- **更新记录**

    *LinearModel.py* : 
    
        2019-05-23更新说明:
            增加功能：
            1. 增加部分约束的Alpha Lasso，允许只对部分系数进行l1惩罚，不加惩罚的系数index用beta_index参数指明
            2. 所有模型增加批训练功能，应对数据量太大而运算资源不够的问题。
                1）批训练的batch_size默认值为1024，如果将batch_size改为X_train.shape[0]，则等价于全量训练
                2）由于批训练每次迭代只训练一部分数据，最大迭代次数max_iter应该相应增大
                3）为了确保模型收敛，请务必在模型训练结束后使用模型的plotLoss()方法查看损失函数变动图
            3. 所有模型class增加说明文档，在jupyter中，可以通过将光标置于model class括号中，按Shift+Tab查看
            >Btw, 今天是个神奇的日子,20190523,从左边起，依次去掉每一个数字，得到的数字都是质数！20190523,190523,90523,523,23,3！
            >所以今天适合写代码！
        2019-06-06更新说明:
            增加功能：
            1. 正系数Lasso回归Positive Lasso。在每一次迭代之前将所有的负系数置0。class名为PosLassoClassifier和PosLassoClassifierCV
                说明：正系数Lasso适用于WOE等系数均为正值的场景。在已知系数为正的情况下，使用正系数Lasso可以更好地对模型进行正则化，
                      增加系数的合理性和模型稳定性。
        2019-07-12更新说明：
            1. 对于LassoClassifier,EnetClassifier,PosLassoClassifier,
                   LassoClassifierCV,EnetClassifierCV,PosLassoClassifierCV 等六个Class，   
		        1）增加Y_hat_track属性，记录模型收敛后100次迭代中训练集上各个样本的Y预测概率值，用于评价收敛之后模型在随机梯度下降中的稳定性。
		        2）增加beta_track属性，记录模型收敛后100次迭代中训练集上各个样本的系数值，用于评价收敛之后模型在随机梯度下降中的稳定性。
		        3）增加beta_mean参数，当beta_mean=True时，系数的估计采取收敛后100次迭代的均值
	        2. 优化LassoClassifierCV,EnetClassifierCV,PosLassoClassifierCV的跟踪显示
	        3. 更改所有模型的迭代步长缩减率，使之更加合理。
        2019-07-17升级说明
            1. 修复predict_proba中的部分bug
        2019-08-06升级说明
            1. 对于LassoClassifier,EnetClassifier,PosLassoClassifier,
                   LassoClassifierCV,EnetClassifierCV,PosLassoClassifierCV 等六个Class，
               增加start_point参数。如果设置为'ridge',则会选取线性岭回归估计量作为系数迭代的初始值
               否则，选取随机正态分布作为系数迭代初始值。根据https://www.tuicool.com/articles/mAbiq2，
               用最小二乘法估计量作为初始值可以有效减少不收敛的发生，加快收敛速度。
        2019-08-08升级说明
            1. 对于LassoClassifier,EnetClassifier,PosLassoClassifier,
                   LassoClassifierCV,EnetClassifierCV,PosLassoClassifierCV 等六个Class，
               增加Q参数。如果设置为一个小于1的非负值,则损失函数计算均值时会除去大于Q quantile的元素
               减少异常值的影响，使得模型更加稳健。
                
    *MonoLogitTrans.py* : 
    
        2019-07-18升级说明
            1. 增加method属性，用于选择转化函数是否是严格单调函数
            2. 增加target属性，用于选择目标变量是连续变量还是二分类变量
            3. 完善transform属性对异常值的处理