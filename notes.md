### Scene Graph Generation by Iterative Message Passing
* Authors: Danfei Xu, Yuke Zhu, Christopher B. Choy, Li Fei-fei. arXiv: 1701.02426
* Reading date: 25/01/2017
* Categories: Computer Vision, Machine Learning
* Tag: Object detection, Sence understanding

In this paper the authors proposed a novel scene graph generation method. Scene graph is a structural representation that captures objects and their semantic relationships, whose value has been proven in a wide range of visual task.  

Based on Region Proposal Network, this work perform approximate inference with GRUs on the complete graph whose nodes are the object proposal boxes. Then the model with produce predictions of the object label of boxes, the bounding box offsets of the boxes and the relationships between all boxes. This procedure utilizes the interaction information between all proposal boxes. The experiments shows that their method outperforms previous state-of-the-art models on Visual Genome and NYU Depth v2 dataset.

### Fully Convolutional Networks for Semantic Segmentation
* Authors: Jonathan Long, Evan Shelhamer, Trevor Darrell. CVPR 2015.
* Reading date: 05/02/2017
* Categories: Computer Vision, Machine Learning
* Tag: Semantic segmentation, Fully convolutional networks.

This paper proposed fully convolutional networks, which are able to take input of arbitrary size. As an extension of convolutional neural networks, FCN highlights the translation invariance property. Since FCN is able to produce pixel-level output, it is quite suitable for semantic segmentation task.

The novel component proposed in this paper is upsampling operation by backward strided convolution. With the help of skip connections from the upsampling feature maps of different strides, FCN achieved a new state-of-the-art semantic segmentation result on Pascal VOC and NYU Depth v2 dataset.

The fantastic ideas of fully convolutional networks includes:
1. Upsampling by backward convolution, which is flexible and computation economical.
2. Skip connections from different strides based on upsampling operation.
