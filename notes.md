### Teaching Compositionality to CNNs
* Authors: Austin Stone, Huayan Wang, Michael Stark, Yi Li, D. Scott Phoenix, Dileep George
* Index: arXiv 1706.04313
* Reading date: 17/06/2017
* Categories: Computer Vision, Machine Learning
* Tag: Network regularization

This work proposed to enhanced state-of-the-art neural networks for computer vision by a novel compositional loss to decouple the relationship of different objects present in the image. With the help of mask annotation, the proposed method encourages the models to focus on the object features rather than the context.

The experiments show the performance of the proposed method. While the community cares more about learning context information, this work shows that decoupling context may helps some computer vision tasks.

### Attention Is All You Need
* Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
* Index: arXiv 1706.03762
* Reading date: 16/06/2017
* Categories: Natural Language Processing, Machine Learning
* Tag: Machine Translation

This work presents a novel machine translation without recurrent or convolutional neural networks, which achieve state-of-the-art results with much fewer parameters and computational resources. The proposed Transformer module is based on multi-head attention, whose kernel is scaled dot-product attention.

The experiments on WMT 2014 datasets show the impressive performance of the proposed method, while the experiment on English constituency parsing shows the generalization ability of the proposed method. More exploration should be done in the future to expolit the potential of the Transformer module.

### On Calibration of Modern Neural Network
* Authors: Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger
* Index: arXiv 1706.04599, ICML 2017
* Reading date: 15/06/2017
* Categories: Machine Learning
* Tag: Confidence calibration

This work surveys the confidence calibration of the softmax output from modern neural networks. The experiment result shows that batch normalization and and increase of model capacity are related to model miscalibration while improve the classification performance. The cause is the overfitting to negative log-likelihood loss. The authors proposed temperature scaling calibration, which is the simplest extension of Platt scaling to solve this problem on large scale dataset problems. The temperature constant is optimized on validation set.

The experiments on various computer vision and natural language processing tasks shows the effectiveness of the proposed method. This work provides the insights of model confidence calibration problem with a simple but effective tool to calibrate existing model, which is important for the tasks that are sensitive to the confidence value.

### Accurate, Large Minibatch SGD: Training Imagenet in 1 Hour
* Authors: Priya Goyal, Piotr Dollar, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, Kaiming He
* Index: arXiv 1706.02677
* Reading date: 13/06/2017
* Categories: Computer Vision, Machine Learning, Parallel Computation
* Tag: Optimization

This technological report presents some important tricks in training neural networks on large scale datasets. The linear scaling rule of learning rates is the key to large minibatches training. Gradual warmup and per-worker batch minibatches are proposed to make faster training feasible. Momentum correction is used in modifying learning while maintaining the performance. Binary blocks algorithm is used in multi-machine training.

The experiments on different settings of Imagenet shows that the proposed methods are able to make large-scale training with large batch size of 8192 while maintains almost same performance.

### Show, Hear and Read: Deep Aligned Representations
* Authors: Yusuf Aytar, Carl Vondrick, Antonio Torralba
* Index: arXiv 1706.00932
* Reading date: 07/06/2017
* Categories: Computer Vision, Machine Learning
* Tag: Representation Learning

This work presents a unified framework to learn cross-modal representations from paired data. The proposed method is based on teacher-student framework, which poses a loss for learning alignment by model transfer. Another ranking-based loss is also proposed.

The experiments on large paired datasets show the proposed beaten the baseline significantly in cross modal retrieval and some other tasks, and the possibility to learn representations of concepts that are independent to modalities.

### What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?
* Authors: Alex Kendall, Yarin Gal
* Index: arXiv 1703.04977
* Reading date: 07/06/2017
* Categories: Computer Vision, Machine Learning
* Tag: Bayesian Neural Networks

This paper presents a novel method to estimate the degree of uncertainty in deep neural networks, which makes it possible to improve previous models. The uncertainty is modeled as the scale parameter of a Gaussian or Lapalacian distribution depending on the input data. The proposed loss function is based on dropout variational inference, which helps efficient training and inference.

The experiments on semantic segmentation and pixel-wise depth regression shows the proposed method is able to improve state-of-the-art models, and the learned uncertainty metric is reasonable, such as the blurred positions and object edges.

### Controllable Invariance through Adversarial Feature Learning
* Authors: Qizhe Xie, Zihang Dai, Yulun Du, Eduard Hovy, Graham Neubig
* Index: arXiv 1705.11122
* Reading date: 01/06/2017
* Categories: Machine Learning
* Tag: Representation Learning

This work presents a method to learn representation with some specific invariance, motivated by adversarial learning. The proposed method contains a discriminator to figure out the specific attributes we hope to eliminate. The encoder tries to confuse the discriminator by producing representation with the desired invariance.

The experiments on multi-lingual datasets and and Extended Yale B image classification datasets show the potential of the proposed method in invariant representation learning tasks.

### Deep Learning is Robust to Massive Label Noise
* Authors: David Rolnick, Andreas Veit, Serge Belongie, Nir Shavit
* Index: arXiv 1705.10694
* Reading date: 31/05/2017
* Categories: Machine Learning
* Tag: Noisy Label

This work investigates the performance of deep neural networks under massive label noise. Their experiments show that convolution networks are able to achieve impressive accuracy when the correct labels are less than 10% of the whole dataset. They proposed that a sufficiently large training set is more important that a lower level of noise. However, when the minimum dataset size required for effective training is satisfied, increasing the dataset further dose not appreciably increase accuracy.

Finally they proposed that larger batch size is able to eliminate the bias introduced by noisy labels. And injecting noise data is reducing the effective batch size, which requires lower learning rate for stable training.

### Dilated Residual Networks
* Authors: Fisher Yu, Vladlen Koltun, Thomas Funkhouser
* Index: arXiv 1705.09914
* Reading date: 31/05/2017
* Categories: Computer Vision, Machine Learning
* Tag: Base model

This work presents a family of residual networks named dilated residual networks, which extending the respective fields by dilation convolutions. In order to get high-frequency activations, they replace max pooling by convolutional filters in ResNet.
xit
The experiments shows that the proposed method improves the baseline model significantly in both classification and localization tasks. The experiment on semantic segmentation also achieves state-of-the-art result.

### Semantically Decomposing the Latent Spaces of Generative Adversarial Networks
* Authors: Chris Donahue, Akshay Blasubramani, Julian McAuley, Zachary C. Lipton
* Index: arXiv 175.07904
* Reading date: 26/05/2017
* Categories: Machine Learning
* Tag: Generative Adversarial Networks

This work presents a new algorithm to decomposing the latent spaces of GAN without specific loss function. The proposed method decomposes the latent space into an identity space, which controls the identity portion of latent codes, and a observation space, which takes effects on the presenting attributes like pose and lighting. A pairwise training scheme is used to make different codes work in different ways.

The experiments on MS-CeleA and Amazon shoes datasets show the effectiveness of the proposed method in building identity latent codes. The proposed architectures sheds lights on controlling the latent spaces of GANs.

### Better Text Understanding Through Image-To-Text Transfer
* Authors: Karol Kurach, Sylvain Gely, Michal, Jastzebski, Philip Haeusser, Olivier Teytaud, Damien Vincent, Olivier Bousquet
* Index: arXiv 1705.08386
* Reading date: 25/05/2017
* Categories: Natural Language Processing, Computer Vision, Machine Learning
* Tag: Word Embedding

This work presents Visual Enhanced Text Embedding, a word embedding method enhanced with image data. With the help of a pretrained image model, VETE tries to connect the image representation and the embedding of its text description. A novel loss based on Pearson correlation is proposed to get better results.

The experiments on various datasets show the good performance on different tasks brought by VETE.  

### SSD: Single Shot MultiBox Detector
* Authors: Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg
* Index: arXiv 1512.02325, ECCV 2016
* Reading date: 17/05/2017
* Categories: Computer Vision, Machine Leanring
* Tag: Object Detection

This work presents Single Shot MultiBox Detector, a fast object detector with high performance. The propose to utilize extra feature layers on different scales to improve the performance of detector. The default boxes and aspect ratios are used to make the regression easier.

With carefully tuning, the proposed method outperforms state-of-the-art detectors on VOC datasets and get comparable result on COCO dataset. The extremely fast speed is attractive to the community.

### Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representation
* Authors: Eliyahu Kiperwasser, Yoav Goldberg
* Index: TACL vol. 4 313-327, 2016
* Reading date: 21/05/2017
* Categories: Natural Language Processing, Machine Learning
* Tag: Dependency Parsing

This work addresses the difficulty of feature engineering in building natural language parsers. The proposed method utilize the bidirectional LSTM to generate word features. The margin based hinge loss is used to train the model in both trainsition-based and graph-based parsers.

The experiments on PTB and CTB shows the proposed methods achieve state-of-the-art without hand-crafted features. This work sheds lights on the research of fully end-to-end RNN-based parsering system, and becomes the standard framework of later parsers.

### Arc-swift: A Novel Transition System for Dependency Parsing
* Authors: Peng Qi, Christopher D. Manning
* Index: arXiv 1705.04434, ACL 2017
* Reading date: 17/05/2017
* Categories: Natural Language Processing, Machine Learning
* Tag: Dependency Parsing

This work presents a novel transition-based dependency parser that takes global information into consider. The non-local transition arc is introduced to predict a constituent that has consumes all of its right dependents efficiently. The proposed method equips the transition-based parser with the ability like graph-based parser.

The experiments show that Arc-swift outperforms previous state-of-the-art transition-based model on EN-UD dataset and gets comparable result on PTB-SD datasets.

### Curiosity-driven Exploration by Self-supervised prediction
* Authors: Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, Trevor Darrell
* Index: arXiv 1705.05363, ICML 2017
* Reading date: 16/05/2017
* Categories: Reinforcement Learning, Machine Learning
* Tag: Exploration

This paper proposed the Intrinsic Curiosity Module, a small sub-network for deep reinforcement learning systems to model the curiosity as an extra reward. The intrinsic curiosity module tries to learn the model's actions and the feature vectors of the environment in next step. Then the L2-distance between the prediction features and the real features are used as a curiosity reward, which encourages the exploration.

The experiments on A3C agents show that the proposed method accelerates the convergence and makes training possible in some difficult setting.

### Unimodal probability distribution for ordinal classification
* Authors: Christopher Beckham, Christopher Pal
* Index: arXiv 1705.05278, ICML 2017
* Reading date: 16/05/2017
* Categories: Machine Learning
* Tag: Ordinal Classification

This paper addresses the ordinal classification problem: in which the prediction targets are discrete and ordered. The Poisson distribution is used in the proposed model to enhance the ordinal prior. The proposed method is able to be packaged as a normal neural layer, so could be used in existing deep learning models.

The experiments show the proposed method is competitive compared to cross-entropy loss, but the prediction distribution is more meaningful.

### SCNet: Learning Semantic Correspondence
* Authors: Kai Han, Rafael S. Rezende, Bumsub Ham, Kwan-Yee K. Wong, Minus Cho, Cordelia Schmid, Jean Ponce
* Index: arXiv 1705.04043
* Reading date: 15/05/2017
* Categories: Computer Vision, Machine Learning
* Tag: Semantic Correspondence

This work addresses the problem of establishing semantic correspondence between two images. The proposed model is a modified from probabilistic Hough matching, in which a sparse geometry kernel is used to model the geometry similarity. A gradient-based learning algorithm is utilized to learn the appearance. Based on state-of-the-art detection network, The matching of regions from different images is performed.

The model outperforms previous state-of-the-art models that based on hand-crafted features. The sparsity of the geometry kernel makes the training possible.

### A Deep Reinforced Model for Abstractive Summarization
* Authors: Romain Paulus, Caiming Xiong, Richard Socher
* Index: arXiv 1705.04304
* Reading date: 15/05/2017
* Categories: Natural Language Processing, Machine Learning
* Tag: Text Summarization

This work presents a RNN-based abstractive summarization model based on intra-attention and reinforcement learning. The intra-temporal attention is added into the encoder and the decoder to overcome the repetition output problem. Some strong constraints to avoid repetition is also utilized. Finally, the self-critical policy gradient is used to combine the ROUGH score into the objective function.

The experiments on NYT dataset and CNN/Daily Mail dataset shows that the proposed loss function and the intra-temporal attention really enhanced the encoder-decoder based abstractive summarization system.

### Unsupervised Object Discovery and Localization in the Wild: Part-based Matching with Bottom-up Region Proposals
* Authors: Minsu Cho, Suha Kwak, Cordelia Schmid, Jean Ponce
* Index: arXiv 1501.06170, CVPR2015
* Reading date: 14/05/2017
* Categories: Computer Vision
* Tag: Object Detection, Object Matching

This paper proposed a part-based matching method to address unsupervised object discovery and localization problems. The authors proposed the Hough match confidence to evaluate the how good a matching pair is. The model utilizes the assumption that decouples the appearance likelihood and the geometry likelihood with a pose displacement parameter. The foreground localization and object discovery algorithm is based on the Hough match confidence.

The experiments on PASCAL VOC datasets show that the proposed method achieve a rather high score among unsupervised methods. The proposed part-based matching algorithm is useful in a lot of different tasks.

### Inferring and Executing Programs for Visual Reasoning
* Authors: Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Judy Hoffman, Li Fei-fei, C. Lawrence Zitnick, Ross Girshick
* Index: arXiv 1705.03633
* Reading date: 11/05/2017
* Categories: Computer Vision, Machine Learning
* Tag: Visual Question Answering

This paper presents a new method for solving visual question answering problems with reasoning requirements. The proposed method utilizes a program generator based on sequence-to-sequence LSTM to predict a expression program for the query. Then an execution engine is utilized to compute the answer. The execution engine is based on general module components based on standard residual blocks.

The proposed method outperforms baselines on CLEVR datasets. This work shows the possibility to use general components in module networks. But more works should be performed for more general VQA tasks.

### Convolutional Sequence to Sequence Learning
* Authors: Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, Yann N. Dauphin
* Index: arXiv 1705.03122
* Reading date: 10/05/2017
* Categories: Natural Language Processing, Machine Learning
* Tag: Machine Translation

This work presents a fully convolutional sequence to sequence model for machine translation. Besides using deep convolution neural networks as the encoder and decoder, the authors proposed multi-step attention and position embeddings to improve the performance.

With carefully initialization and normalization, the model outperforms previous state-of-the-art machine translation models based on recurrent neural networks in experiments. This work shows the potential of CNN in NLP is not fully explored yet.

### Metacontrol For Adaptive Imagination-Based Optimization
* Authors: Jessica B. Hamrick, Andrew, J. Ballard, Razvan, Pascanu, Oriol Vinyals, Nicolas Heess, Peter W. Battaglia
* Index: arXiv 1705.02670
* Reading date: 09/05/2017
* Categories: Reinforcement Learning, Machine Learning
* Tag: Meta-learner

This work shows the possibility to incorporate computation cost as a part of the loss function in the model training. The proposed method uses a reinforcement learning agent as the meta-controller to manage the inference procedure by choosing which expert models to run. The meta-controller is trained with a loss combined prediction loss and the computation cost. The expert models and the real controller are trained as the normal neural networks and reinforcement learning agents. While the meta-controller is trained by the REINFORCE algorithm.

The experiments on Space Ship dataset show the effectiveness of this method. However, more experiments should be executed on different tasks.

### Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks
* Authors: Samy Bengio, Oriol Vinyals, Navdeep Jaitly, Noam Shazeer
* Index: arXiv 1506.03099, NIPS 2015
* Reading date: 09/05/2017
* Categories: Machine Learning
* Tag: Recurrent Neural Networks

This work proposed curriculum learning, a scheduled ground truth sampling method, to improve the performance of recurrent neural network in sequence prediction tasks. In order to overcome the discrepancy between the training and the inference of the recurrent neural networks caused by the feeding of the previous token, the authors proposed to use token predicted from the models with a probability, rather simply use the ground truth previous token.

The experiments on MSCOCO caption, constituency parsing and speech recognition show the proposed method is effective and general to most sequence learning tasks by using recurrent neural networks.

### Sequential Attention
* Authors: Sebastian Brarda, Philip Yeres, Samuel R. Bowman
* Index: arXiv 1705.02269
* Reading date: 08/05/2017
* Categories: Natural Language Processing, Machine Learning
* Tag: Attention Mechanism, Text Comprehension

This paper introduces sequential attention, a sequential version of traditional attention layers. Compared to traditional attention, the proposed method used a recursive neural network to model the attention vector, which takes the correlations on attention of different words into considering.

The experiments on CNN dataset and Who did What dataset show the effectiveness of this new attention mechanism. More experiments should be performed to explore the properties of this method.

### Xception: Deep Learning with Depthwise Separable Convolutions
* Authors: Francois Chollet
* Index: arXiv 1610.02357
* Reading date: 07/05/2017
* Categories: Computer Vision, Machine Learning
* Tag: Base model

This paper extend the Inception hypothesis and proposed a new building block for deep neural networks: Depthwise Separable Convolutions. Depthwise separable convolutions are composed by a depthwise convolution, followed by a pointwise convolution. This architecture explicitly factors the convolution operation into an operation considering spatial correlations and an operation for cross-channel correlations.

The experiments showed that with same amount of parameters, the proposed method outperforms previous Inception networks. The depthwise separable convolutions show itself as a desirable building blocks for computer vision.

### Discourse-Based Objectives for Fast Unsupervised Sentence Representation Learning
* Authors: Yacine Jernite, Samuel R. Bowman, David Songtag
* Index: arXiv 1705.00557
* Reading date: 02/05/2017
* Categories: Natural Language Processing
* Tag: Sentence Representation

This paper introduced three objective loss functions for unsupervised sentence representation learning through discourse coherence. The objectives are binary ordering of sentences, next sentence relations and conjunction predictions. The experiments shows that the proposed method achieve competitive results with much fewer training time.

### Automatic Real-time Background Cut for Portrait Videos
* Authors: Xiangyong Shen, Ruixing Wang, Hengshuang Zhao, Jiaya Jia
* Index: arXiv 1704.08812
* Reading date: 02/05/2017
* Categories: Computer Vision
* Tag: Background Cut

This paper proposed a novel background cut algorithm that is fast and accurate. Based on a light ResNet, a background attenuation model is introduced. The manually cropped background samples are also pushed into the network. The concatenation of the features extracted from the input frames and the background samples are used to predict a segmentation maps. Finally, a spatial-temporal refinement network is used to improve the quality of the final results.

The proposed method is attractive in commercial applications. The experiments show that the proposed method outperforms the baseline significantly. This paper shows the effectiveness of background sample features in background cut.

### Abstract Syntax Networks for Code Generation and Semantic Parsing
* Authors: Maxim Rabinovich, Mitchell Stern, Dan Klein
* Index: arXiv 1704.07535, ACL 2017
* Reading date: 01/05/2017
* Categories: Natural Language Processing, Machine Learning
* Tag: Semantic Parsing, Code Generation

The authors introduce abstract syntax networks, which is a neural network model framework based on abstract syntax trees. The model is of encoder-decoder style. The encoder is a normal recurrent neural network like LSTMs, while the decoder is structured as a collection of mutually recursive modules. Since there are composite types, constructors, constructor fields and primitive types in the abstract syntax trees, four different models are proposed to process these structures.

The experiments on code generation show that the proposed method improve the state-of-the-art result significantly. The experiments on semantic parsing shows that the method is also competitive in natural language parsing.

### ScaleNet: Guiding Object Proposal Generation in Supermarkets and Beyond
* Authors: Siyuan Qiao, Wei Shen, Weichao Qiu, Chenxi Liu, Alan Yuille
* Index: arXiv 1704.06752
* Reading date: 25/04/2017
* Categories: Computer Vision
* Tag: Object Detection

This paper proposed ScaleNet, which improves object proposals by predicting the scales in the input images. The scale distribution learning is performed by minimizing Kullback-Leibler divergence to the ground truth scales distribution. The images are resized to some scales sampled from the scale distribution.

The experiments on COCO dataset and Amazon Supermarket dataset showed that the proposed method outperforms baseline models in average recall.

### Detecting and Recognizing Human-Object Interactions
* Authors: Georgia Gkioxari, Ross Girshick, Piotr Dollar, Kaiming He
* Index: arXiv 1704.07333
* Reading date: 25/04/2017
* Categories: Computer Vison
* Tag: Object Detection, Visual Relationship

This paper proposed InteractNet, a model designed for detecting human-object interactions. Based on Faster-RCNN framework, a novel target localization model is proposed to measure the possibility of the interactions with the image feature of human and the position of the object.

The experiments performed on V-COCO datasets show that the proposed method beats the baseline with a large margin.

### Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data
* Authors: Nicolas Papernot, Martin Abadi, Ulfar Elingsson, Ian Goodfellow, Kunal Talwar
* Index: ICLR 2017
* Reading date: 24/04/2017
* Categories: Machine Learning
* Tag: Privacy-preserving Learning

This paper introduce Private Aggregation of Teacher Ensembles, a privacy-preserving learning methods. The framework is to train an ensemble of teacher models on the sensitive data and use the teacher models as an interface for student models. The training of student model is possible to access the knowledge of sensitive data without touching them.

A GAN-based semi-supervised learning model is used as the student model. By the theoretical inductions and experiments, the authors show that the proposed method achieve high performance with good protection of sensitive data.  

### Improved Techniques for Training GANs
* Authors: Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki, Cheung, Alec Radford, Xi Chen
* Index: arXiv 1606.03498
* Reading date: 21/04/2017
* Categories: Machine Learning
* Tag: Generative Adversarial Networks

This paper proposed some important tricks to improve the training of GANs based on the observations of the training procedure of GANs. The proposed feature matching tries to constraint the solution space of GANs. While other methods are able to make the generator stable and diverse. A semi-supervised learning method is also proposed to utilize the labeled data.

The experiments show the quality improvements in the generated images. While the proposed tricks lead us to get a better understand of GANs.

### Learning to Reason: End-to-End Module Networks for Visual Question Answering
* Authors: Ronghang Hu, JAcob Andreas, Marcus Rohrbach, Trevor Darrell, Kate Saenko
* Index: arXiv 1704.05526
* Reading date: 21/04/2017
* Categories: Computer Vision, Machine Learning
* Tag: Visual Question Answering, Reinforcement Learning

This paper proposed End-to-End Module Networks, which is a class of models that are capable of predicting novel network construction directly from textual input. With a properly designed functional expression system, the model predicts an expression for an image with a sequence-to-sequence RNN. Reverse Polish Notation is used to simplify the learning task. Policy gradient is used to make end-to-end learning possible. They also proposed a method to clone behavior of expression prediction from expert polices.

The experiments show that their method achieve state-of-the-art result on two synthetic visual question answering datasets.

### Neural Module Networks
* Authors: Jacob Andreas, Marcus Rohrbach, Trevor Darrell, Dan Klein
* Index: arXiv 1511.02799, CVPR 2016
* Reading date: 20/04/2017
* Categories: Computer Vision, Natural Language Processing, Machine Learning
* Tag: Visual Question Answering

This paper proposed Neural Module Networks, which are a kind of models that are composed by some small sharing modules. In order to solve visual QA task, they designed five different modules: "FIND", "TRANSFORM", "COMBINE", "DESCRIBE", "MEASURE". Then a parser is used to transform the question into a symbolic representation, which guides the construction of the network. Finally, the network is used to solve the problem.

The experiments showed that the proposed achieves state-of-the-art performance on both synthetic and natural datasets. The most important contribution of this paper is the possibility to learn some small modules and compose them dynamically.

### Improving Object Detection With One Line of Code
* Authors: Navaneeth Bodla, Bharat Singh, Rama Chellappa, Larry S. Davis
* Index: arXiv 1704.04503
* Reading date: 18/04/2017
* Categories: Computer Vision
* Tag: Object Detection, Non-maximum suppression

This paper proposed an extension of traditional NMS algorithm as Soft-NMS. The proposed algorithm suppresses the low-score proposal boxes by decreasing their scores rather than simply removing them. The authors proposed two different functions to change the scores. The experiments show that the proposed method improves state-of-the-art object detection by at least 1% in AP. Since Soft-NMS is easy to implement in current system architecture, it could be seen as a competitive alternative of NMS.

### Get To The Point: Summarization with Pointer-Generator Networks
* Authors: Abigail See, Peter J. Liu, Christopher D. Manning
* Index: arXiv 1704.04368, ACL 2017
* Reading date: 17/04/2017
* Categories: Natural Language, Machine Learning
* Tag: Text Summarization, Pointer Networks

This paper addresses the text summarization problem with a pointer-generator network. Based on sequence to sequence models, the proposed utilize a pointer mechanism to decide whether to generate a new word from vocabulary, or copy from original text. A coverage mechanism is also proposed to penalize repeatedly attending to the same locations.

The experiments show that the proposed model outperforms baseline with at least 2 points improvements in ROUGH scores. The generated summarization is more natural and diverse compared to the previous systems. But generating high-level abstractions remains a open question.

### Spatial Memory for Context Reasoning in Object Detection
* Authors: Xinlei Chen, Abhinav Gupta
* Index: arXiv 1704.04224
* Reading date: 14/04/2017
* Categories: Computer Vision, Machine Learning
* Tag: Object Detection

In this paper, the authors proposed to utilize a spatial memory module to improve object detection framework by providing the ability of context reasoning. The proposed spatial memory is based on 2D recurrent neural network with an ROI indexing. The convolution features are fused before the final predictions.

The experiments show that this method improves the precision of the prediction a lot. But the recall drops. The fancy idea is that it is possible to reason by fusing the features before ROI Pooling.

### Weight Uncertainty in Neural Networks
* Authors: Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, Daan Wierstra
* Index: arXiv 1505.05424, ICML 2015
* Reading date: 11/04/2017
* Categories: Machine Learning
* Tag: Bayesian Neural Networks

This paper introduce Bayes by Backprop, a novel and efficient algorithm that allows to optimize neural networks with complex prior weights distribution. Firstly, the authors proved the unbiased Monte Carlo gradients which make the backpropagation possible. With a Gaussian variational posterior as an example, they proposed scale mixture Gaussian prior, which avoids the overfitting of the poor initial parameters.

 The experiments on Bandits, MNIST digits classification task and simple non-linear regression show that efficiency of the proposed method in making reasonable predictions about unseen data.

### A Hierarchical Approach for Generating Descriptive Image Paragraphs
* Authors: Jonathan Krause, Justin Johnson, Ranjay Krishna, Li Fei-fei
* Index: arXiv 1611.06607, CVPR 2017
* Reading date: 10/04/2017
* Categories: Computer Vision, Natural Language Processing, Machine Learning
* Tag: Image Caption

This paper proposed to overcome the limitations of basic image caption task by generate descriptive paragraphs for the input images. In order to solve the challenge task, they proposed a model based on region detector and hierarchical recurrent neural networks. A projection pooling is used to extract features of all object regions into a fixed size vector. The sentence RNN gets the feature vector as input and produce topic vectors as the inputs of the following word RNN.

The experiments show that this model outperforms the baseline models. But there is still a large gap between the model and humans.

### Not All Pixels Are Equal: Difficulty-Aware Semantic Segmentation via Deep Layer Cascade
* Authors: Xiaoxiao Li, Ziwei Liu, Ping Luo, Cheng Change Loy, Xiaoou Tang
* Index: arXiv 1704.01344, CVPR 2017
* Reading date: 06/04/2017
* Categories: Computer Vision, Machine Learning
* Tag: Semantic Segmentation

This paper presents Deep Layer Cascade, a novel method to improve existing semantic segmentation model both in performance and speed. The idea is to output the easy mask pixels in advance, which makes later layers focus on the hard pixels. By combining the outputs of different stages, the final segmentation is able to beat the baseline models.

The model used in the experiments is based on Inception-ResNet, which contains enough stages. The experiments on VOC 2012 and Cityscapes showed that the proposed model achieves the state-of-the-art level.

### Semantic Instance Segmentation via deep Metric Learning
* Authors: Alireza Fathi, Zbigniew Wojna, Vivek Rathod, Peng Wang, Hyun Oh Song, Sergio Guadarrama, Kevin P. Murphy
* Index: arXiv 1703.10277
* Reading date: 05/04/2017
* Categories: Computer Vision, Machine Learning
* Tag: Instance Segmentation

This paper proposed a metric learning method to solve semantic instance segmentation problem. The authors found that it is possible to learn a similarity metric of different instances in the image that is able to distinguish different instances even when they are in same category. Based on the successful embedding metric model, they proposed a seediness model that is used to generate seed of mask, which is similar to a semantic segmentation classification model.

The experiments show that this model is a state-of-the-art proposal-free model for instance segmentation problem. The result is also competitive to the proposal-based models on PASCAL VOC dataset.

### DeNet: Scalable Real-time Object Detection with Directed Sparse Sampling
* Authors: Lachlan Tychsen-Smith, Lars Petersson
* Index: arXiv 1703.10295
* Reading date: 31/03/2017
* Categories: Computer Vision, Machine Learningp
* Tag: Object detection

This paper proposed DeNet, a real-time object detector based on estimating the corner distribution and sparse sampling. The estimation of corner distribution is produced by a deconvolution network, which follows the fully convolutional network structure. By matching two corners, the model is able to predict the probability of each bounding box.

The experiments showed that the model outperforms previous state-of-the-art real-time object detectors and achieves a comparable result to best detectors. The novel proposals are the most important contribution of this paper.

### Objects as context for part detection
* Authors: Abel Gonzalez-Garcia, Davide Modolo, Vittorio Ferrari
* Index: arXiv 1703.09529
* Reading date: 29/03/2017
* Categories: Computer Vision, Machine Learning
* Tag: Object detection, Object part detection

This paper addresses the problem of semantic part detection. By incorporating the information of object detection boxes, the model is able to refine part detection result. The proposed method to get information combined is to use the feature vectors of the object appearance and the part appearance. Relative location information that extracted by an offset net is also added into the combined feature vector.

The experiments on PASCAL-part dataset showed that the proposed method is able to improved an existing object detection model significantly in part detection task. The experiments on fine-grained recognition dataset CUB200-2011 also showed the proposed method gets a state-of-the-art result.

### Evolution Strategies as a Scalable Alternative to Reinforcement Learning
* Authors: Tim Sailmans, Jonathan Ho, Xi Chen, Ilya Sutskever
* Index: arXiv 1703.03864
* Reading dazte: 28/03/2017
* Categories: Reinforcement Learning, Evolution Algorithms, Machine Learning
* Tag: Gradient-free optimization

This paper presents an scalable evolution strategies algorithm that is able to replace policy gradient methods for reinforcement learning. They found that virtual batch normalization is key to the success of evolution strategies. An antithetic sampler is used to reduce the variance of the perturbation. The evolution strategies with Gaussian distribution is in face a zero-order gradient estimation, which could produce an informative gradient estimation.

The experiments show that their proposed method is able to achieve comparable result using only CPUs. The scalability is an attractive feature of this method. The experiments also show that gradient-free optimization is viable to solve reinforcement learning problems.

### Linguistic Knowledge as Memory for Recurrent Neural Networks
* Authors: Bhuwan Dhingra, Zhilin Yang, William W. Cohen, Ruslan Salakhutdinov
* Index: arXiv 1703.02620
* Reading date: 27/03/2017
* Categories: Natural Language Processing, Machine Learning
* Tag: Recurrent Neural Networks

This paper presents MAGE-GRU, which is a model that is able to utilize linguistic knowledge provided by other tools to improve learning the long range dependencies in RNN. MAGE-GRU is an extension of GRU that is able to work on directed acyclic graph. By explicitly decomposing parameters for different edge types, the model is able to learn specific effects for different linguistic relation.

The experiments on QA tasks based on Stanford NLP tools show that MAGE-GRU outperforms previous state-of-the-art QA systems impressively. Beside the huge improvements they have got on QA tasks, the framework is also a great contribution.

### Mask R-CNN
* Authors: Kaiming He, Georgia Gkioxari, Piotr Dollar, Ross Girshick
* Index: arXiv 1703.06870
* Reading date: 21/03/2017
* Categories: Computer Vision, Machine Learning
* Tag: Instance segmentation, Object detection

This work presented Mask R-CNN, a R-CNN variants that is able to output instance segmentation. The key of this work is RoIAlign, which is a quantization-free layer that faithfully preserves exact spatial locations compared to RoIPooling. RoIAlign is able to replace all RoIPooling in previous models. This architecture also decouples the mask and the class predictions.

The experiments on COCO datasets is impressive. The single model achieves state-of-the-art performance in instance segmentation and object detection tasks. The model also gains a large improvements in human pose estimation task.

### DeViSE: A Deep Visual-Semantic Embedding Model
* Authors: Andrea Frome, Greg S. Corrado, Jonathon Shlens, Samy Bengio, Jeffrey Dean, Mar'Aurelio Ranzato, Tomas Mikolov
* Index: NIPS 2013
* Reading date: 20/03/2017
* Categories: Computer Vision, Natural Language Processing, Machine Learning
* Tag: Word embedding, Zero-shot Learning

This paper presented a deep visual-semantic embedding model that is able to identify visual objects using labeled images and unannotated text. The model is based on state-of-the-art image classification model and word embedding model. A new hinge rank loss is proposed to incorporate information from both sources.  

The experiments show that this model is able to achieve the performance of state-of-the-art softmax baseline, while be able to generalize to unseen labels. This paper shows the possibility to utilize language prior to solve zero-shot learning problems.

### Understanding Black-box Predictions via Influence Functions
* Authors: Peng Wei Koh, Percy Liang
* Index: arXiv 1703.04730
* Reading date: 16/03/2017
* Categories: Machine Learning
* Tag: Understanding learning system

This paper proposed an approximation method of influence functions to understand the behavior of a learning system by finding the influence of each training example. This method is able to find the outliers that changed the learning system behaviors, debug domain mismatch and so on.

The experiments showed that the HVP approximation is able to compute an accurate estimation of the influence functions. And as this method provided a tool to understand the learning system, a lot of experiments show that it is able to provide some explainations of the prediction of the neural networks.

### Deep and Hierarchical Implicit Models
* Authors: Dustin Tran, Rajesh Ranganath, David M. Blei
* Index: arXiv 1702.08896
* Reading date: 14/03/2017
* Categories: Machine Learning
* Tag: Implicit probabilistic models

This paper developed deep implicit models and hierarchical models. These models are two classes of Bayesian hierarchical models which only assume a process that generate samples. Variational inference is used to optimize the models. However, the local densities $p(x_n, z_n| beta)$ and its variational approximation are bot intractable if the assumption of the latent models are removed, which poses a challenge to build the model. The authors proposed a ratio estimation for the ratio of log-likelihoods of two intractable densities, which is inspired by GAN.

The experiments show that this framework is expressive and expends Bayesian analysis to a lot of new ares.

### Deep Value Networks Learn to Evaluate and Iteratively Refine Structured Outputs
* Authors: Michael Gygli, Mohammad Norouzi, Anelia Angelova
* Index: arXiv 1703.04363
* Reading date: 14/03/2017
* Categories: Machine Learning
* Tag: Structured learning

This paper proposed a new framework for structured learning problems. The authors proposed to use a network to learn the evaluation criterion directly, which is inspired by the value network in deep reinforcement learning. In the inference part, a gradient descending algorithm is used to find the output that minimizes the output of value network. In the training part, the value network is trained with a special loss function and some adversarial tuples.

The experiments show that this framework is able to achieve or outperforms previous state-of-the-art methods on different structured learning tasks covering multi-label classification and image segmentation.

### Semantic Object Parsing with Graph LSTM
* Authros: Xiaodan Liang, Xiaohui Shen, Jiashi Feng, Liang Lin, Shuicheng Yan
* Index: ECCV 2016, arXiv 1603.07063
* Reading date: 10/03/2017
* Categories: Computer Vision, Machine Learning
* Tag: Semantic parsing

This work proposed Graph LSTM, which is able to replace CRF in existing computer vision task. The information propagation is based on the operations of LSTM. Compared to existing LSTM variants, Graph LSTM is more general and more suitable for semantic parsing task. The adaptive forget gate used in this model which make the nodes pay different attention on different neighbors is a new idea for LSTM design.

The experiments are performed on 4 challenging semantic parsing task, and show that Graph LSTM-based outperforms previous models significantly in all tasks.

### Learning from Noisy Labels with Distillation
* Authors: Yuncheng Li, Jianchao Yang, Yale Song, Liangliang Cao, Jiebo Luo, Jia Li
* Index: arXiv 1703.02391
* Reading date: 09/03/2017
* Categories: Computer Vision, Machine Learning
* Tag: Noisy data learning, Distillation

This paper shows the possibility that learn from a large noisy dataset with a small cleaned dataset. While the noisy dataset is easier to get, this method is quite useful in dealing with current large-scale video dataset. While the rationale of this method is not so clear, the experiments show that this method outperforms baseline and some regularization methods.

The key to this method is the hyperparameter that balances the information from noisy data and existing good classifier trained on a clean dataset. They also proposed to gain more information from existing knowledge graph. However, the experiments show that it does not improve the model significantly.

### Tree-Structured Reinforcement Learning for Sequential Object Localization
* Authors: Zequn Jie, Xiaodan Liang, Jiashi Feng, Xiaojie Jin, Wen Feng Lu, Shuicheng Yan
* Index: NIPS 2016, arXiv 1703.02710
* Reading date: 09/03/2017
* Categories: Computer Vision, Machine Learning
* Tag: Reinforcement learning

This paper proposed a better region proposal model based on reinforcement learning. The procedure of region proposal is treated as a sequential decision process. The model is optimized to generate better region proposals with two type of operations on the existing proposals. The also proposed to perform tree based search to explore more operations. The reward function is carefully designed to encourage the model to generate better region proposals as soon as possible.

The experiments shows that their region proposal model beats existing state-of-the-art models and gains comparable object detection performance with R-CNN with less region proposals.

### Axiomatic Attribution for Deep Networks
* Authors: Mukund Sundararajan, Ankur Taly, Qiqi Yan
* Index: arXiv 1703.01365
* Reading date: 07/03/2017
* Categories: Machine Learning
* Tag: Understanding learning system

This paper proposed two axioms of attribution of predictions of deep neural networks, i.e. figuring out the importance of all inputs to the final predictions. The axioms are sensitivity and implementation invariance. With these two properties, they proposed Integrated Gradients, the first network attribution method that satisfies the two axioms. They prove that the method is solid in theory.

In the experiment part, the authors performed 5 different experiments on 3 different type of task. The experiments cover computer vision, natural language processing and chemistry model. The experiments showed that the proposed method is a better network visualization tools compared to previous methods in practice.

### A Structure Self-attentive Sentence Embedding
* Authors: Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, Yoshua Bengio
* Index: ICLR 2017
* Reading date: 07/03/2017
* Categories: Natural Language Processing, Machine Learning
* Tag: Representation learning, Sentence embedding

This paper proposed a framework to learn structured sentence embedding by self attention mechanism. The embedding of a sentence is a matrix in fact. Different rows of the matrix is a "channel" of the information. The authors proposed a special penalization term to encourage different attention vectors to focus different aspect of the input information, which increase the diversity of the attention vectors.

This method also provided a fantastic way to visualize which part of sentence is more important in the final result. The experiments on the Age dataset, the Yelp dataset and the Stanford Natural Language Inference Corpus shows that this sentence embedding is able to achieve state-of-the-art performance in different natural language processing task with a general framework.

### How to Escape Saddle Points Efficiently
* Authros: Chi Jin, Rong Ge, Praneeth Netrapalli, Sham M. Kakade, Michael I. Jordan
* Index: arXiv 1703.00887
* Reading date: 06/03/2017
* Categories: Machine Learning, Optimization
* Tag: Non-convex optimization

This paper presents a simple algorithm, Perturbed Gradient Descent, that is able to escape saddle point based on the mere information of gradients. Since the algorithm is a modified version of SGD, it is possible to be used in training deep neural networks. The basic idea of this algorithm is that when the gradient is too small, a random noise is added into the parameters. The authors showed that with proper hyperparameters, the algorithm is able to escape saddle points in limited steps with nearly no extra resources.

### You Only Look Once: Unified, Real-Time Object Detection
* Authors: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
* Index: CVPR 2016, arXiv 1506.02640
* Reading date: 03/03/2017
* Categories: Computer Vision, Machine Learning
* Tag: Object detection

This paper proposed YOLO object detector, a region-based object detector that only processes the image once. This object detector is able to achieve competitive  performance with much higher speed that support real-time detection. The model produces region proposal with a region proposal network and builds a class probability map at the same time. The class probability map is used to tell what object is in the region proposal, which is produced by a fully convolutional network. The experiments show that this detector is the state-of-the-art real-time object detector.

### Stacked Hourglass Networks for Human Pose estimation
* Authors: Alejandro Newell, Kaiyu Yang and Jia Deng
* Index: ECCV 2016, arXiv 1603.06937
* Reading date: 03/03/2017
* Categories: Computer Vision, Machine Learning
* Tag: Pose estimation, Hourglass network

This paper introduce to use a new network architecture, stacked hourglass networks, for human pose estimation. This network is built with skip connection between the down-sampling features and the up-sampling features whose resolution is same. It provided a method to produce high-resolution and high-level features.

Based on hourglass network, the authors proposed intermediate supervision in the stacked hourglass network.The experiments show that stacked "short" hourglass networks with intermediate supervision is able to achieve the best performance. To conclude, the most important contribution of this paper is the stacked hourglass network architecture.
### The Statistical Recurrent Unit
* Authors: Junier B. Oliva, Barnabas Poczos, Jeff Schneider
* Index: arXiv 1703.00381
* Reading date: 02/03/2017
* Categories: Recurrent Neural Network, Machine Learning
* Tag: Moving average

This paper proposed a gate-free recurrent unit based on the moving average of statistics of sequence data. It is inspired by mean map embeddings. The value of this work is that it provided a new view of recurrent neural network and gate mechanism.

The experiments on synthetic data shows that SRU outperforms LSTM and GRU a lot. While the experiments on different real temporal data shows that SRU is really good at capturing long term dependencies.

### Improving Object Detection with Region Similarity Learning
* Authors: Feng Gao, Yihang Lou, Yan Bai, Shiqi Wang, Tiejun Huang ,Ling-Yu Duan
* Index: arXiv 1702.00234
* Reading date: 02/03/2017
* Categories: Computer Vision, Machine Learning
* Tag: Object detection, Embedding learning

This paper proposed triplet embedding to incorporate the constraint of relative similarity distances between positives and negatives into region-based detector learning. By carefully sampling hard negatives, the triplet embedding distances work as a regularizer in the loss function.

The experiments on VOC07+12 shows that with this method the detector gain a lot improvement in performance.

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
* Categories: Computer Vision, Machine Learning
* Tag: Semantic segmentation

This paper proposed Progressively Diffused Network, which is able to diffuse information in the locally features in the convolutional feature maps. After diffusing information, the feature maps contain more global information while maintaining spatial information.

In this work, the diffusing layer is implemented by convolutional LSTM. The experiments showed that this method may be able to improve existing semantic segmentation models.  

### Understanding Convolution for Semantic Segmentation
* Authors: Panqu Wang, Pengfei Chen, Ye Yuan, Ding Liu, Zehua Huang, Xiaodi Hou, Carrison Cottrell
* Index: arXiv 1702.08502
* Reading date: 01/03/2017
* Categories: Computer Vision
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
* Categories: Natural Language Processing, Machine Learning
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
* Categories: Network Compression, Machine Learning
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
