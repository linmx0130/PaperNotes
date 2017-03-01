### Spatially Adaptive Computation Time for Residual Networks
* Authors: Michael Figurnov, MAxwell D. Collins, Yukun Zhu, Li Zhang, Jonathan Huang, Dmitry Vetrov, Ruslan Salakhutdinov
* Index: arXiv 1612.02297
* Reading date: 02/03/2017
* Categories: Computer Vision, Machine Learning
* Tag: Adaptive computation time, Attention mechanism

Inspired by adaptive computation time mechanism in recurrent neural networks, the authors proposed adaptive computation time and spatially adaptive computation time for deep residual networks. The network produce a halting score at the end of each residual block. When the cumulative halting score is large enough, the computation is stopped.

The novelty of this work is that the authors proposed a ponder score which is a diffierentiable upper bound on the number of evaluated units N. By minimize the ponder score, the network is able to find a suitable time to stop. They also proposed to use convolution to perform spatially adaptive computation time. The experiments on two large-scale object detection show that this method get a rather good trade-off between the computation time and the performance. It could be seen as a special attention mechanism too, while the halting score is the value of the feature maps. The weighted sum of feature maps is used to produce final result.

### Beyond Skip Connections: Top-Down Modulation for Object Detection
* Authors: Abhinav Shrivastava, Rahul Sukthankar, Jitendra Malik, Abhinav Gupta
* Index: arXiv 1612.06851
* Reading date: 01/03/2017
* Categories: Computer Vision, Machine Learning
* Tag: Object detection, Convolutional neural networks

This paper proposed top-down modulation that connects features from down-sampling convolution and up-sampling convolution layers. This network provided a simple and natural way to produce better high resolution feature maps.

The experiments show that this method is useful in object detection task. The nice property of this method is that it could be used in any tasks that requires good high resolution feature maps.

### Progressively Diffused Networks for Semantic Image Segmentation
* Ruimao Zhang, Wei Yang, Zhangli Peng, Xiaogang Wang, Liang Lin
* Index: arXiv 1702.05839
* Reading date: 01/03/2017
* Categoires: Computer Vision, Machine Learning
* Tag: Semantic segmentation

This paper proposed Progressively Diffused Network, which is able to diffuse information in the locally features in the convolutional feature maps. After diffusing information, the feature maps contain more global information while maintaining spatial information.

In this work, the diffusing layer is implemented by convolutional LSTM. The experiments showed that this method may be able to improve existing semantic segmentation models.  

### Understanding Convolution for Semantic Segmentation
* Authors: Panqu Wang, Pengfei Chen, Ye Yuan, Ding Liu, Zehua Huang, Xiaodi Hou, Carrison Cottrell
* Index: arXiv 1702.08502
* Reading date: 01/03/2017
* Categoires: Computer Vision
* Tag: Semantic Segmentation

This paper proposed two tools to improve convolution-based semantic segmentation systems: dense upsampling convolution and hybrid dilated convolution.

The key idea of dense upsampling convolution is to add more parameters for predicting more pixels with the final feature maps. While hybrid dilated convolution is trying to solve the problem that dilated convolution may lose some features. It also increase the respective field of the feature maps. The experiments on Cityscapes, KITTI and Pascal VOC2012 show that these methods are useful.

### VisualBackProp: visualizing CNNs for autonomous driving
* Authors: Mariusz Bojarski, Anna Choromanska, Krzysztof Choromanski, Bernhard Firner, Larry Jackel, Urs Muller, Karol Zieba.
* Index: arXiv 1611.05418
* Reading date: 28/02/2017
* Categories: Machine Learning
* Tag: Network visualizing

This paper introduce a simple but novel method to visualize the decision reason of a convolutional neural network. The method simply take the mean value of all channels in the output of a layer as the visualization mask of this layer. The deeper and smaller visualization mask is then scaled up to suit the shape of previous layer. The scaled up visualization mask is used to multiply the feature map of previous layer pointwise.

The experiments show that their method is accurate and fast enough for understanding and debugging the deep neural networks. The theoretical analysis of the output this method is able to identify the contribution to the final predictions of input pixels.

### Understanding Deep Learning Requires Rethinking Generalization
* Authors: Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals
* Index: ICLR 2017
* Reading date: 28/02/2017
* Categories: Machine Learning
* Tag: Deep learning theory

This paper poses a lot of new question about existing theory on deep learning and machine learning. Their experiments show two important facts: 1. The neural network *can* learn by remembering all training samples. 2. The explicit regularization cannot explain the generalization ability of deep neural networks.

The authors proposed that stochastic gradient descending is able to converge to a model with small norm in most time. It is natural to assume that SGD provided an implicit regularization. More research should be taken to help us to understand the generalization ability of neural networks.

### ViP-CNN: A Visual Phrase Reasoning Convolutional Neural Network for Visual Relationship Detection
* Authors: Yikang Li, Wanli Ouyang, Xiaogang Wang
* Index: arXiv 1702.07191
* Reading date: 27/02/2017
* Categories: Computer Vision, Machine Learning
* Tag: Visual relationship, Object detection

This paper introduce ViP-CNN to deal with the visual relationship detection. Compared to previous methods that performs object detection as first stage, this work produce triplet proposal that contains the subject, the object and the whole phrase at the same time. With these proposals, the modules later are able to perform information passing ("reasoning structure" in the paper) inside the network.

Their experiments show the ability of this structure outperforms previous state-of-the-art models. The idea that reduce the amount of stages to gain more interaction information is the key insight of this paper.

### Learning Chained Deep Features and Classifiers for Cascade in Object Detection
* Authors: Wanli Ouyang, Ku Wang, Xin Zhou, Xiaogang Wang
* Index: arXiv 1702.07054
* Reading date: 25/02/2017
* Categories: Computer Vision, Machine Learning
* Tags: Object Detection

This paper introduce a novel method to build cascade model for object detection. The key of their methods contains two points: 1. Features in different stages should be diverse, which can be achieved by padding context into ROI. 2. Using the result of previous stage as a prior, which make the later stages focus on the failure cases of previous stage. This method seems a special way of boosting.

The experiments showed that their methods improves the model significantly.

### Snapshot Ensembles: Train 1, get M for free
* Authors: Gao Huang, Yixuan Li, Geoff Pleiss, Zhuang Liu, John E. Hopcroft, Kilian Q. Weinberger
* Index: ICLR 2017
* Reading date: 23/02/2017
* Categories: Ensemble Models, Machine Learning
* Tags: Cyclic LR schedule

This paper introduce a novel model ensembling method based on cyclic learning rate scheduling. Cyclic learning rate is able to help the deep networks escape from a local minima and fall into another "better" local minia. The authors proposed to use the ensemble of the snapshots of all models in the local minima.

Their experiments show that without increasing training cost, snapshot ensembling is able to get a high performance ensamble model. They also shows that the diversity of parameters in snapshots is the key of this successful method.

### PixelNet: Representation of the pixels, by the pixels, and for the pixels.
* Authors: Aayush Bansal, Xinlei Chen, Bryan Russell, Abhinav Gupta, Deva Ramanan
* Index: arXiv 1702.06506
* Reading date: 22/02/2017
* Categories: Computer Vision, Machine Learning
* Tag: Fully convolutional networks

The authros demonstrate that stratified sampling of pixels is able to strengthen existing fully convoluional architectures on several different pixel-level tasks.  With hypercolumn in PixelNet, it is possible to perform sparse predictions, which accelerate the training speed.

The experiments on semantic segmentation, surface normal estimation and edge detection shows that the proposed method is general enough for different tasks.

### Person Search with Natural Language Description
* Authors: Shuang Li, Tong Xiao, Hongsheng Li, Bolei Zhou, Dayu Yue, Xiaogang Wang
* Index: arXiv 1702.05729
* Reading date: 22/02/2017
* Categories: Computer Vision, Natural Language Processing, Information Retrieval
* Tag: Person retrieval

This paper proposed a new task that searching persons in large-scale image databases with natural language description. They built a large-scale benchmark for this task. The experiments on humans show that human is able to get better retrieval results with natural language rather than attributes.

In order to solve this task, the authors also proposed an innovative Recurrent Neural Network with Gated Neural Attention mechanism, which combines the state-of-the-art models in NLP and CV areas.

### Optimization as a Model for Few-Shot Learning
* Authors: Sachin Ravi, Hugo Larochelle
* Index: ICLR 2017
* Reading date: 21/02/2017
* Categories: Few-Shot Learning, Machine Learning
* Tag: Meta Learning

This paper is based on *Learn to learn by gradient descent by gradient descent(arXiv 1606.04474)*. The authros extends the methods to meta learning, which achieves a great success in few-shot learning.

By finding that the rule of LSTM cell unit is similar to the update rules of gradient descending algorithms, they proposed to use a LSTM to learn the procedure of "learning from few instances". With careful parameter sharing and preprocessing, the meta-learner is able to treat the parameters of the learner as its cell. The experiments on Mini-Imagenet shows that their method is able to outperform previous state-of-the-art few-shot learning methods.

### Learning to Detect Human-Object Interactions
* Authors: Yu-Wei Chao, Yunfan Liu, Xieyang Liu, Huayi Zeng, Jia Deng
* Index: arXiv 1702.05448
* Reading date: 20/02/2017
* Categories: Computer Vision, Machine Learning
* Tag: Object detection, Action detection

This paper introduce a new dataset on detecting human-object interactions with a model based on object detectors. Since the dataset provides the bounding boxes of human and objects, it is possible to detect interactions based on the results of object detections.

They also proposed to construct interaction patterns for human-object pair to improve the performance of interaction detections

### Training Region-based Object Detectors with Online Hard Example Mining
* Authors: Abhinav Shrivastava, Abhinav Gupta, Ross Grishick
* Index: CVPR 2016, arXiv 1604.03540
* Reading date: 19/02/2017
* Categories: Computer Vision, Machine Learning
* Tag: Object detection

This paper proposed to use Online Hard Example Mining(OHEM) to boost the performance of a region-based object detector. With the training of region-based object detectors, samples exposed to the RCNN become easier and easier to the detectors. The authros utilize a special bootstrap method to mine "hard examples".

The experiments showed that OHEM is cost reasonable and improves state-of-the-art detectors surprisingly.

### Featrue Pyramid Networks for Object Detection
* Authors: Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie
* Index: arXiv 1612.03144
* Reading date: 18/02/2017
* Categories: Computer Vision
* Tag: Object detection, Convolutional networks

This paper proposed feature pyramid networks, which combine ideas of hand-engineered features and neural networks. Tasks like object detection and semantic segmentation requires high-resolution features, which cannot be satisfied with features from the top convolutional networks. So the authors proposed to gain higher resolution features by adding top-down pathway with lateral connections and upsampling. It is natural to perform such operations in fully convolutional architectures like residual networks.

Their experiments with Fast/Faster RCNN and segmentation proposals show that the proposed method is a general way to improve existing detection and segmentation models based on convolutional networks.

### Frustratingly Short Attention Spans in Neural Language Modeling
* Authors: Michal Daniluk, Tim Rocktaschel, Johannes Welbl, Sebastian Riedel
* Index: ICLR 2017, arXiv 1702.04521
* Reading date: 16/02/2017
* Categoires: Natural Language Processing, Machine Learning
* Tag: Attention mechanism, Language model

This paper introduce key-value-predict attention as an extension of attention mechanism in RNN language model. By spliting the hidden vector into three different part for three different functions, their attention argumented neural language model achieved a great success.

Their experiments show that the neural language models  utilized a memory of only the most recent history and failed to exploit long-range dependencies. It seems that in language modeling task, it is too difficult to utlized long-range information.

### Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized models
* Authors: Sergey Ioffe
* Index: arXiv 1702.03275
* Reading date: 15/02/2017
* Categories: Deep Learning, Machine Learning
* Tag: Batch normalization.

Batch normalization is a successful trick in training deep neural network. However, it makes an assumption that samples in each minibatch is independent, which does not hold in practice. When the batch size is small, batchnorm would even harm the network. This paper proposed Batch Renormalization to overcome this problem.

By introducing the difference parameters between minibatch statistics and moving averages, batch renormalization overcomes the problem led by the incoherent in the training and inference of batch norm. The experiments showed that Batch Renormalization is helpful to improve the network with small-size minibatches and non-i.i.d. minibatches.

### Incremental Network Quantization: Twoard Lossless CNNs with Low-Precision Weights
* Authors: Aojun Zhou, Anbang Yao, Yiwen Guo, Lin Xu, Yurong Chen
* Index: ICLR 2017, arXiv 1702.03044
* Reading date: 14/02/2017
* Categoires: Network Compression, Machine Learning
* Tag: Network compression

While deep learning models contains a large amount of parameters, they are difficult to use in mobile devices or embedding systems. This paper proposed Incremental Network Quantization, which is able to convert a pretrained model into low-bit representation with better performence.

In order to utilize specific hardware or FPGA, the authors decide to quantize the float point numbers into the power of 2. With a incremental quantization strategy, the network will be partially retrained after being partially quantized. The experiments showed that this method is able to produce higher performence models with rather low precision of parameters compared to full-presicion models. The compression ratio is more than 50 times.

### Automatic Rule Extration From Long Short Term Memory Networks
* Authors: W. James Murdosh, Arthur Szlam
* Index: ICLR 2017, arXiv 1702.02540
* Reading date: 13/02/2017
* Categories: Recurrent Neural Network, Machine Learning
* Tag: LSTM

This paper proposed a method to exploit rule patterns from a trained LSTM network to show how LSTM works. The authors provided an additive decomposition of the LSTM cell, which suggests a natural definition of the "importance score" of a input term to the whole sequence.

While this score can be used to visualize the network, it can also be used to find the rule-based patterns that determine the final output. The authors perform experiments to extract rule patterns from the network to build a pattern matcing classifier. This experiment showed that the rules are really main part of the information that LSTM learned.

### Deep Learning with Dynamic Computation Graphs
* Authors: Moshe Looks, Marcello Herreshoff, DeLesley Hutchins, Peter Norvig
* Index: ICLR 2017
* Reading date: 09/02/2017
* Categories: Machine Learning
* Tag: Machine learning library

This paper proposed to use dynamic batching algorithm on a dynamic computation graph to accelerate the training of deep learning model with complicate structures like tree.

At the same time they also provided a combinator library for neural networks based on Tensorflow, which simplifies the coding of building dynamic computation graph. The library is inspired by functional programming language, which is suitable for describing combination structures.

### End-to-end People Detection in Crowded Scenes
* Authors: Russel Stewart, Mykhaylo Andriluka, Andrew Y. Ng
* Index: CVPR 2016
* Reading date: 09/02/2017
* Categories: Computer Vision, Machine Learning
* Tag: Object detection

This work tried to solve people detection in crowded scenes by replacing NMS with a LSTM network in faster RCNN framework. In order to train the LSTM to produce boxes in order of the descending confidence, the authors proposed a special Hungarian loss, which is based on Hungarian algorithm. The loss is elaborate and diffierentiable almost everywhere.

The idea of the proposed loss is to treat boxes combination problem as a *graph minimum-cut problem*. In the testing part, the detector should stitch boxes from different blocks grabbed from the original data. It also performs a graph minimum-cut rather than NMS.

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

### Scene Graph Generation by Iterative Message Passing
* Authors: Danfei Xu, Yuke Zhu, Christopher B. Choy, Li Fei-fei.
* Index: arXiv: 1701.02426
* Reading date: 25/01/2017
* Categories: Computer Vision, Machine Learning
* Tag: Object detection, Sence understanding

In this paper the authors proposed a novel scene graph generation method. Scene graph is a structural representation that captures objects and their semantic relationships, whose value has been proven in a wide range of visual task.  

Based on Region Proposal Network, this work perform approximate inference with GRUs on the complete graph whose nodes are the object proposal boxes. Then the model with produce predictions of the object label of boxes, the bounding box offsets of the boxes and the relationships between all boxes. This procedure utilizes the interaction information between all proposal boxes. The experiments shows that their method outperforms previous state-of-the-art models on Visual Genome and NYU Depth v2 dataset.
