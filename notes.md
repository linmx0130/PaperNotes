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
