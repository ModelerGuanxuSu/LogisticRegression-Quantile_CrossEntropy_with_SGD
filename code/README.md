
*MonoLogitTrans.py* : 

    Description:
        The module offers an algorithm of single varaible transformation, which has following propertities:
            1) positively corelated with P(Y=1)
            2) offers paramater to choose if the transformation is guaranteed to be monotone
            3) theoretically equivalent to the logit of P(Y=1) 
        
        How:
            If the parameter method='wide':
                Fit MLP between single varaible and Y. The loss function is made up of two parts,
                cross entropy and loss of monotonicity.
                The difination of loss of monotonicity is:
                    $\text{loss_monotonous}(x,f(x)) = 1 - |(\rho(x,f(x)))|$
                    $\rho(x,y)$ is Pearson correlation coefficient.
                The final loss function is:
                    loss = cross_entropy($\sigma$(f(x)),y)+lambda_monotonous*loss_monotonous(x,f(x))
            If the parameter method='strict':
                The sign of all the weights in the same hiden layer are constrainted to be the same. Hence, the 
                MLP is nested function of monotone functions, which means that the MLP is a monotone function.
                In this situation, the loss function is simply the cross entropu loss.
    Version info：
        sklearn 0.20.1
        tensorflow 1.13.1
        python 3.7.1
        
*LinearModel.py* : 

    Description:
        The module offers a siries of class of logistic regression, which are similar to the logistic regressions offered
        by sklearn when people use them. For example, they have methods like fit(), predict(), predict_proba(), etc.
        The main modifications are:
            1）In order to cooperate with single variable transformations like WOE which are positively correlated with
               P(Y=1), the class PosLassoClassifier and PosLassoClassifierCV offers logistic regression with constraint 
               that all the coefficients are positive.
            2）They adopt SGD with choice of start point of ridge regression estimator or random normal.
            3) To deal with outliers, the loss of each batch can exclude the largest (1-Q0)*100% elements with label
               Y=0 and (1-Q1)*100% elements with label Y=1, before taking the mean, which makes the model nonsensitive
               of cost.
            4) The final estimator can be set to be the mean of estimated values of 100 iterations after converge, 
               in order to get a robust estimation and variable selection, which makes the model nonsensitive to 
               randomness of sampling.
    Version info：
        sklearn 0.20.1
        tensorflow 1.13.1
        matplotlib 3.0.2
        python 3.7.1
