# Pose-guided action recognition pytorch

This is a work on exploring the use of human pose information in helping action recognition. For complete report, check [here](https://drive.google.com/file/d/1slIjLgRIqRsZwTR9ohVr61Y7OR2oFGdO/view?usp=sharing).

Implemented three different models: 
- attention-pooling method (with ResNet101 backbone)
- C3D method 
- two-stream method

## Will human pose information help with action recognition
First, we explored whether the use of human pose information would help in action recognition tasks.

Pose information is applied by weakening background using mask information in C3D architecture and directly adding joint position heat-map into top-down attention in attention-pooling method.

The results from both methods show slight improvement with human pose information:

With and without pose information in C3D architecture:

<p align="center">
<img src="https://pic-markdown.s3.amazonaws.com/region=us-west-2&tab=overview/2018-05-27-044341.png"/>
</p>

With and without pose information in attention-pooling architecture:

<p align="center">
<img src="https://pic-markdown.s3.amazonaws.com/region=us-west-2&tab=overview/2018-05-27-044642.png"/>
</p>

## Towards properly using pose information in action recognition
Since the human pose information does help with action recognition, then we need to figure out the proper way of using the pose information. 

We modified two-stream architecture, and replaced the optical-flow stream with VGG16, the intuition is, by using VGG16, we could use a network pre-trained on ImageNet, instead of having the entire optical-flow stream trained from scratch, using VGG16 will also improve efficiency compared to using optical flow.

The problem is how to properly use the ImageNet pre-trained model, since the number of channel for temporal stream input is 3 * F (F being the frame number), and the ImageNet pre-trained model has input of 3 channels. We compared the results get from not loading the first layer of pre-trained model, and the results obtained by initializing the weights of first layer by replicating the weights of first layer F times. We use pre-trained ResNet101 for spatial stream and AlexNet (works better than VGG16 according to our experiments) for temporal stream. And as can be seen from the results, replicating the weights of first layer by F times greatly helped the network to converge faster and get to a better optima.

<p align="center">
<img src="https://pic-markdown.s3.amazonaws.com/region=us-west-2&tab=overview/2018-05-27-045441.png" />
</p>

Then we also tried using the puppet mask to incorporate pose information in two-stream architecture, we weaken the context to help network learn to detect the change in pose. But as can be seen from the results, weakening background doesn’t work very well in two-stream architecture.

<p align="center">
<img src="https://pic-markdown.s3.amazonaws.com/region=us-west-2&tab=overview/2018-05-27-050106.png" width=600 height=300/>
</p>

In conclusion, the human pose information actually help with action recognition, and we improved both the accuracy and efficiency of the two-streamed architecture by replacing the optical-flow stream with ImageNet-pretrained VGG16.

## Authorship

Below is the authorship information for this project.

  * __Author__:  Shangwu Yao
  * __Email__:   shangwuyao@gmail.com

Copyright (C) 2018, Shangwu Yao. All rights reserved.
