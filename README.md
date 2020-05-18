# PyTorch for Deep Learning and Computer Vision

This repository contains worked out implementations for [PyTorch for Deep Learning and Computer Vision](https://www.udemy.com/course/pytorch-for-deep-learning-and-computer-vision/?LSNPUBID=QhjctqYUCD0&ranEAID=QhjctqYUCD0&ranMID=39197&ranSiteID=QhjctqYUCD0-1hYZOGjDH3dISFFHX6uK7g).

Topics covered:
1. Linear Regression

2. Logistic Regression

3. Perceptron based shallow Neural-Network

   <img src="https://miro.medium.com/max/520/1*UGSVZ_Xq19x_wQeVlz76dQ.jpeg" alt="Perceptron" style="zoom: 50%;" />

4. Deep Neural Networks

5. Image Recognition using Convolution Neural Network

- MNSIT dataset

    

  ​	<img src="https://miro.medium.com/max/584/1*2lSjt9YKJn9sxK7DSeGDyw.jpeg" alt="MNSIT" style="zoom:50%;" />

  

- CIFAR10 dataset

  

  ​	<img src="https://miro.medium.com/max/964/1*syyml8q8s1Yt-iEea5m1Ag.png" alt="CIFAR10" style="zoom: 33%;" />

  

6. Transfer Learning

- **Alex Net**: 
  
    AlexNet is the name of a convolutional neural network (CNN), designed by Alex Krizhevsky, and published with Ilya Sutskever and Krizhevsky's doctoral advisor Geoffrey Hinton.

    AlexNet competed in the ImageNet Large Scale Visual Recognition Challenge on September 30, 2012. The network achieved a top-5 error of 15.3%, more than 10.8 percentage points lower than that of the runner up. The original paper's primary result was that the depth of the model was essential for its high performance, which was computationally expensive, but made feasible due to the utilization of graphics processing units (GPUs) during training.
    
	 <img src="https://engmrk.com/wp-content/uploads/2018/10/AlexNet_Original_Image.jpg" alt="AlexNet" style="zoom:80%;" />
    [AlexNet: The Architecture that Challenged CNNs - Medium](https://towardsdatascience.com/alexnet-the-architecture-that-challenged-cnns-e406d5297951)



- **VGG16**: The input to the network is image of dimensions *(224, 224, 3)*. The first two layers have *64* channels of *3\*3* filter size and same padding. Then after a max pool layer of stride *(2, 2)*, two layers which have convolution layers of 256 filter size and filter size *(3, 3)*. This followed by a max pooling layer of stride *(2, 2)* which is same as previous layer. Then there are *2* convolution layers of filter size *(3, 3)* and *256* filter. After that there are *2* sets of *3* convolution layer and a max pool layer. Each have *512* filters of *(3, 3)* size with same padding.This image is then passed to the stack of two  convolution layers. In these convolution and max pooling layers, the  filters we use is of the size *3\*3* instead of *11\*11* in AlexNet and *7\*7* in ZF-Net. In some of the layers, it also uses *1\*1* pixel which is used to manipulate the number of input channels. There is a padding of *1-pixel* (same padding) done after each convolution layer to prevent the spatial feature of the image.

    After the stack of convolution and max-pooling layer, we got a *(7, 7, 512)* feature map. We flatten this output to make it a *(1, 25088)* feature vector.After this there are *3 fully* connected layer, the first layer takes input from the last feature vector and outputs a *(1, 4096)* vector, second layer also outputs a vector of size *(1, 4096)* but the third layer output a *1000* channels for *1000* classes of ILSVRC challenge, then after the output of 3rd fully  connected layer is passed to softmax layer in order to normalize the  classification vector. After the output of classification vector top-5  categories for evaluation. All the hidden layers use ReLU as its  activation function. ReLU is more computationally efficient because it  results in faster learning and it also decreases the likelihood of  vanishing gradient problem.  

    <img src="https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png" alt="VGG16" style="zoom:67%;" />

	[VGG16 – Convolutional Network for Classification and Detection](https://neurohive.io/en/popular-networks/vgg16/)