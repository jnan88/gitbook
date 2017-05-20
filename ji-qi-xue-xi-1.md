## 实践工具

* [TensorFlow Playground](http://playground.tensorflow.org/) 提供可视化界面的样本分类
* [ConvNetJS](http://cs.stanford.edu/people/karpathy/convnetjs/) 用于数字和图像识别
* [Keras.js Demo](https://transcranial.github.io/keras-js/) 在浏览器中可视化展现和使用网络模型
* [Anaconda环境](https://www.continuum.io/downloads) 
* [Keras](https://keras.io/#installation)

## 数学基础

* 向量、矩阵和多维数组；
* [卷积运算](http://setosa.io/ev/image-kernels/)提取局部特征；
* 激活函数：[sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function),[tanh](https://www.wolframalpha.com/input/?i=tanh[x])或者[ReLU](https://en.wikipedia.org/wiki/Rectifier_%28neural_networks%29)等；
* [softmax](https://en.wikipedia.org/wiki/Softmax_function)将向量转化为概率值；
* [log-loss\(cross-entropy\)](http://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks)作为惩罚项
* 网络参数优化的梯度[反向传播](http://cs231n.github.io/optimization-2/)算法
* 随机梯度下降及其变种（比如[冲量](http://distill.pub/2017/momentum/)）

## 数学工具

1. 向量计算：[word2vec](http://p.migdal.pl/2017/04/30/p.migdal.pl/2017/01/06/king-man-woman-queen-why.html)

* J. Ström, K. Åström, 和 T. Akenine-Möller 编写的[《Immersive Linear Algebra》](http://immersivemath.com/ila/index.html)
* 应用数学和机器学习基础：《深度学习》的[线性代数](http://www.deeplearningbook.org/)章节
* Brendan Fortuner 的[《Linear algebra cheat sheet for deep learning 》](https://medium.com/towards-data-science/linear-algebra-cheat-sheet-for-deep-learning-cd67aba4526c)

   2. [Numpy](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html) 基础：

* Nicolas P. Rougier 的[《From Python to Numpy》](http://www.labri.fr/perso/nrougier/from-python-to-numpy/)
* [《SciPy lectures: The NumPy array object》](http://www.scipy-lectures.org/intro/numpy/array_object.html)



## 框架

Tensorflow、Theano、Torch和Caffe

[Keras](https://keras.io/) 它属于神经网络的上层封装库，对Tensorflow和Theano做了封装。查看序列模型网络内部数据流的[ASCII summary](https://github.com/stared/keras-sequential-ascii)，比model.summary\(\)用起来更方便。它可以显示层级、数据维度以及待优化的参数数量

相关资料：

Valerio Maggio的[《基于Keras和Tensorflow的深度学习》](https://github.com/leriomaggio/deep-learning-keras-tensorflow)

Erik Reppel写的 [基于Keras 和 Cats 的卷计算机网络可视化](https://hackernoon.com/visualizing-parts-of-convolutional-neural-networks-using-keras-and-cats-5cc01b214e59)

Petar Veličković 写的 [深度学习完全入门：基于Keras的卷计算机网络](https://cambridgespark.com/content/tutorials/convolutional-neural-networks-with-keras/index.html)

Jason Brownlee写的 [用Keras和卷计算机网络识别手写数字](http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)



