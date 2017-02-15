### Scene Graph Generation by Iterative Message Passing
* Authors: Danfei Xu, Yuke Zhu, Christopher B. Choy, Li Fei-fei.
* Index: arXiv: 1701.02426
* Reading date: 25/01/2017
* Categories: Computer Vision, Machine Learning
* Tag: Object detection, Sence understanding

In this paper the authors proposed a novel scene graph generation method. Scene graph is a structural representation that captures objects and their semantic relationships, whose value has been proven in a wide range of visual task.  

Based on Region Proposal Network, this work perform approximate inference with GRUs on the complete graph whose nodes are the object proposal boxes. Then the model with produce predictions of the object label of boxes, the bounding box offsets of the boxes and the relationships between all boxes. This procedure utilizes the interaction information between all proposal boxes. The experiments shows that their method outperforms previous state-of-the-art models on Visual Genome and NYU Depth v2 dataset.

### Fully Convolutional Networks for Semantic Segmentation
* Authors: Jonathan Long, Evan Shelhamer, Trevor Darrell.
* Index: CVPR 2015.
* Reading date: 05/02/2017
* Categories: Computer Vision, Machine Learning
* Tag: Semantic segmentation, Fully convolutional networks

This paper proposed fully convolutional networks, which are able to take input of arbitrary size. As an extension of convolutional neural networks, FCN highlights the translation invariance property. Since FCN is able to produce pixel-level output, it is quite suitable for semantic segmentation task.

The novel component proposed in this paper is upsampling operation by backward strided convolution. With the help of skip connections from the upsampling feature maps of different strides, FCN achieved a new state-of-the-art semantic segmentation result on Pascal VOC and NYU Depth v2 dataset.

The fantastic ideas of fully convolutional networks includes:
1. Upsampling by backward convolution, which is flexible and computation economical.
2. Skip connections from different strides based on upsampling operation.

### Instance-sensitive Fully Convolutional networks
* Authors: Jifeng Dai, Kaiming He, Yi Li, Shaoqing Ren, Jian Sun.
* Index: arXiv: 1603.08678
* Reading date: 06/02/2017
* Categories: Computer Vision, Machine Learning
* Tag: Semantic segmentation, Fully convolutional networks

This work proposed instance-sensitive fully convolutional network, which is an improved version of fully convolutional networks. It produce an instance-sensitive score maps on the top of convolutional feature maps. Each output in the maps is *a classifier of relative positions of instances*, which is the input of an assembling module.

While maintaining the advantages of fully convolutional networks, instance-sensitive FCN is able to distinguish different close instances of same category. Finally, the experiments on PASCAL VOC 2012 and MS COCO dataset shows that their models beat previous state-of-the-art models.

 ### Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
 * Authors: Alec Radford, Luke Metz, Soumith Chintala.
 * Index arXiv:1511.06434
 * Reading date: 06/02/2017
 * Categories: Generative Models, Machine Learning
 * Tag: Generative adversarial networks, Representation learning

 This work proposed an architecture guidelines for stable Deep Convolutional GANs based on experiments. They found that fractional-strided convolutions are always better for generator compared to pooling layers. And fully connected hidden layers should be removed.

 They also tried to "walk" on the mainfold of the latent space and manipulate the generator representation. They found that on the faces dataset, it is possible to get representation of a concept by averaging some samples of the face images following this concept. The arithmetic operation on these vector is able to produce expecting latent representation. This experiment shows that GAN really learns a meaningful representation.

### End-to-end People Detection in Crowded Scenes
* Authors: Russel Stewart, Mykhaylo Andriluka, Andrew Y. Ng
* Index: CVPR 2016
* Reading date: 09/02/2017
* Categories: Computer Vision, Machine Learning
* Tag: Object detection

This work tried to solve people detection in crowded scenes by replacing NMS with a LSTM network in faster RCNN framework. In order to train the LSTM to produce boxes in order of the descending confidence, the authors proposed a special Hungarian loss, which is based on Hungarian algorithm. The loss is elaborate and diffierentiable almost everywhere.

The idea of the proposed loss is to treat boxes combination problem as a *graph minimum-cut problem*. In the testing part, the detector should stitch boxes from different blocks grabbed from the original data. It also performs a graph minimum-cut rather than NMS.

### Deep Learning with Dynamic Computation Graphs
* Authors: Moshe Looks, Marcello Herreshoff, DeLesley Hutchins, Peter Norvig
* Index: ICLR 2017
* Reading date: 09/02/2017
* Categories: Machine Learning
* Tag: Machine learning library

This paper proposed to use dynamic batching algorithm on a dynamic computation graph to accelerate the training of deep learning model with complicate structures like tree.

At the same time they also provided a combinator library for neural networks based on Tensorflow, which simplifies the coding of building dynamic computation graph. The library is inspired by functional programming language, which is suitable for describing combination structures.

### Automatic Rule Extration From Long Short Term Memory Networks
* Authors: W. James Murdosh, Arthur Szlam
* Index: ICLR 2017, arXiv 1702.02540
* Reading date: 13/02/2017
* Categories: Recurrent Neural Network, Machine Learning
* Tag: LSTM

This paper proposed a method to exploit rule patterns from a trained LSTM network to show how LSTM works. The authors provided an additive decomposition of the LSTM cell, which suggests a natural definition of the "importance score" of a input term to the whole sequence.

While this score can be used to visualize the network, it can also be used to find the rule-based patterns that determine the final output. The authors perform experiments to extract rule patterns from the network to build a pattern matcing classifier. This experiment showed that the rules are really main part of the information that LSTM learned.

### Incremental Network Quantization: Twoard Lossless CNNs with Low-Precision Weights
* Authors: Aojun Zhou, Anbang Yao, Yiwen Guo, Lin Xu, Yurong Chen
* Index: ICLR 2017, arXiv 1702.03044
* Reading date: 14/02/2017
* Categoires: Network Compression, Machine Learning
* Tag: Network compression

While deep learning models contains a large amount of parameters, they are difficult to use in mobile devices or embedding systems. This paper proposed Incremental Network Quantization, which is able to convert a pretrained model into low-bit representation with better performence.

In order to utilize specific hardware or FPGA, the authors decide to quantize the float point numbers into the power of 2. With a incremental quantization strategy, the network will be partially retrained after being partially quantized. The experiments showed that this method is able to produce higher performence models with rather low precision of parameters compared to full-presicion models. The compression ratio is more than 50 times.

### Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized models
* Authors: Sergey Ioffe
* Index: arXiv 1702.03275
* Reading date: 15/02/2017
* Categories: Deep Learning, Machine Learning
* Tag: Batch normalization.

Batch normalization is a successful trick in training deep neural network. However, it makes an assumption that samples in each minibatch is independent, which does not hold in practice. When the batch size is small, batchnorm would even harm the network. This paper proposed Batch Renormalization to overcome this problem.

By introducing the difference parameters between minibatch statistics and moving averages, batch renormalization overcomes the problem led by the incoherent in the training and inference of batch norm. The experiments showed that Batch Renormalization is helpful to improve the network with small-size minibatches and non-i.i.d. minibatches.
