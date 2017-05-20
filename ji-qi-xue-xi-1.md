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

2. J. Ström, K. Åström, 和 T. Akenine-Möller 编写的[《Immersive Linear Algebra》](http://immersivemath.com/ila/index.html)

3. 应用数学和机器学习基础：《深度学习》的[线性代数](http://www.deeplearningbook.org/)章节

4. Brendan Fortuner 的[《Linear algebra cheat sheet for deep learning 》](https://medium.com/towards-data-science/linear-algebra-cheat-sheet-for-deep-learning-cd67aba4526c)

   1. [Numpy](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html) 基础：

5. Nicolas P. Rougier 的[《From Python to Numpy》](http://www.labri.fr/perso/nrougier/from-python-to-numpy/)

6. [《SciPy lectures: The NumPy array object》](http://www.scipy-lectures.org/intro/numpy/array_object.html)

## 框架

Tensorflow、Theano、Torch和Caffe

### Keras

[Keras](https://keras.io/) 它属于神经网络的上层封装库，对Tensorflow和Theano做了封装。Keras插件：查看序列模型网络内部数据流的[ASCII summary](https://github.com/stared/keras-sequential-ascii)，比model.summary\(\)用起来更方便。它可以显示层级、数据维度以及待优化的参数数量。

Keras相关资料：

Valerio Maggio的[《基于Keras和Tensorflow的深度学习》](https://github.com/leriomaggio/deep-learning-keras-tensorflow)

Erik Reppel写的 [基于Keras 和 Cats 的卷计算机网络可视化](https://hackernoon.com/visualizing-parts-of-convolutional-neural-networks-using-keras-and-cats-5cc01b214e59)

Petar Veličković 写的 [深度学习完全入门：基于Keras的卷计算机网络](https://cambridgespark.com/content/tutorials/convolutional-neural-networks-with-keras/index.html)

Jason Brownlee写的 [用Keras和卷计算机网络识别手写数字](http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)

### Tensorflow

比Keras更底层、更灵活，能直接对各个多维数组参数做优化。

相关资源：

* 官方教程[Tensorflow Tutorial](https://www.tensorflow.org/versions/master/tutorials/index.html)
* Martin Görner 的[《不读博士也能学习Tensorflow和深度学习》](https://cloud.google.com/blog/big-data/2017/01/learn-tensorflow-and-deep-learning-without-a-phd)
* Aymeric Damien 写的[《Tensorflow入门教程和示例》](https://github.com/aymericdamien/TensorFlow-Examples/)
* Nathan Lintz 写的[《Tensorflow框架简单教程》](https://github.com/nlintz/TensorFlow-Tutorials)

另外，[TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)是一款在训练过程中调试和查看数据非常方便的工具。

## 数据集

[Kaggle](https://www.kaggle.com/)

[MNIST](http://yann.lecun.com/exdb/mnist/)是一份手写数字识别数据集（60000张28x28的灰度图）。它适合用来测试本机上安装的Keras是否成功。

[notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html)（异形字体的字母A-J）

[CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) 是经典的图像识别数据集，都是32x32尺寸的照片。它分为两种版本：10类简单的照片（包括猫、狗、青蛙、飞机等）和100类更难的照片（包括海狸、海豚、水獭、海豹、鲸鱼等）。

深度学习算法都需要大量的数据。如果大家想从头开始训练网络模型，至少需要大约10000张低分辨率的图片。当数训练据匮乏时，网络模型很可能学不到任何模式。那么该怎么办呢？

* 只要肉眼看得清，使用低分辨率的图像也无妨
* 尽可能多的收集训练数据，最好达到百万级别
* 在已有的模型基础上开始训练
* 用现有数据集构造更多的训练数据（比如旋转、平移和扭曲）

## 网络结构

图像领域的几种其它的网络结构：

* [U-Net：用卷计神经网络对生物医学图像进行分割](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) 
  * [卷积神经网络提取视网膜血管](https://github.com/orobix/retina-unet)，Keras实现
  * [Kaggle超声神经提取比赛的深度学习教程](https://github.com/jocicmarko/ultrasound-nerve-segmentation)，Keras实现
* [一种艺术风格的神经算法](https://arxiv.org/abs/1508.06576) ： [神经网络实现风格转换和涂鸦](https://github.com/titu1994/Neural-Style-Transfer)，作者是Somshubra Majumdar

* [图像分割领域的CNN进化史：从 R-CNN 到蒙板 R-CNN](https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4)，作者是Dhruv Parthasarathy



