# PyTorch-and-Monai-for-AI-Healthcare-Imaging

Introduction to Monai, PyTorch, and U-Net.

The U-Net confers to a convolutional network architecture for fast and precise segmentation of images. Recall, a convolutional neural network (CNN or ConvNet) is a network 
architecture for deep learning that learns directly from data. CNNs are particularly useful for finding patterns in images to recognize objects, classes and categorites.
U-Net is developed for the task of semantic segmentation; when a neural network is fed images as inputs, we can choose to classify objects either generally or by instances. 

![image](https://github.com/aliamrod/PyTorch-and-Monai-for-AI-Healthcare-Imaging/assets/62684338/82fe8a8b-0c2a-4faf-bca9-d1cdd7d5454d)


We can consequently predict what object is in the image (image classification), where all the objects are located (image localization/semantic segmentation), or where
individual objects are located (object detection/instance segmentation). _Figure 02_ below demonstates differences between these computer vision tasks. To simplify 
the matter, we only consider classification for only one class and one label (binary classification). 


![image](https://github.com/aliamrod/PyTorch-and-Monai-for-AI-Healthcare-Imaging/assets/62684338/b685165b-7d3f-46ad-999b-cd47b2e8fc6a)






References
[1] "U-Net: Convolutional Networks for Biomedical Image Segmentation". _International Conference on Medical image computing and computer-assisted intervention._
[2] "Understanding U-Net: U-Net has become the go-to method for image segmentation. But how did it come to be?"
[3] "Fully convolutional networks for semantic segmentation." _Proceedings of the IEEE conference on computer vision and pattern recognition._ 
[4] "V-net: Fully convolutional neural networks for volumetric medical image segmentation." _2016 Fourth Internation Conference on 3D Vision (3DV)._
