# Easy Logistic

<p align="center">
    <a href="https://github.com/python/cpython"><img src="https://img.shields.io/badge/Python-3.7-FF1493.svg"></a>
    <a href="https://github.com/tensorflow/tensorflow"><img src="https://img.shields.io/badge/TensorFlow-1.13.1-blue"></a>
    <a href="https://opensource.org/licenses/mit-license.php"><img src="https://badges.frapsoft.com/os/mit/mit.svg"></a>
    <a href="https://github.com/ModelerGuanxuSu/EasyLogistic/raw/master/easylogistic-1.0.1.tar.gz"><img src="https://img.shields.io/badge/downloads-21k-green"></a>
    
</p>



In many financial situations like default prediction, interpretable models are required. Linear models like 
logistic model are often used to reach the requirement. Meanwhile, in order to make the model robust, people
often apply single variable transformation like WOE. However, such transformation has two main drawbacks:

    1) It is sensitive to noise and sometimes yields transformed boxes which are not monotone.
    2) Because of the loss of monotonicity, interpretibility can not be guaranteed.
    
This python module introduce a new method of single variable transformation, `easylogistic.MonoLogitTrans`, which can ensure that the transformation
is monotone, continues, as well as captures the nonlinear relationship between single variable and the log odds of P(Y=1).

The module also offers a series of modified logistic models, which are more robust when the data contains outliers or sparse. The dome jupyter file shows that the modified methods outperforms the state of art logistic regression in terms of accuracy 
and robustness.

## Install

- Download [easylogistic-1.0.1.tar.gz](https://github.com/ModelerGuanxuSu/EasyLogistic/raw/master/easylogistic-1.0.1.tar.gz)
- run `python setup.py install` on commend line


## Owner:

Guanxu Su [Linkedln](https://www.linkedin.com/in/%E5%86%A0%E6%97%AD-%E8%8B%8F-281638147) 