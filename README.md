# Segmentation Using PyTorch and Monai for AI Healthcare Imaging

Overview
-------------------------------------------------
In the following repository, you will locate Python files required to accomplish liver segmentation with Monai and PyTorch. The same code may be utilized against other organs to perform segmentation as well. 

![image](https://github.com/aliamrod/PyTorch-and-Monai-for-AI-Healthcare-Imaging/assets/62684338/d2f132e5-09c0-42bc-96fb-0d4bb19c067f)

### Introduction to Monai, PyTorch, and U-Net.
-------------------------------------------------
The U-Net confers to a convolutional network architecture for fast and precise segmentation of images. Recall, a convolutional neural network (CNN or ConvNet) is a network 
architecture for deep learning that learns directly from data. CNNs are particularly useful for finding patterns in images to recognize objects, classes and categorites.
U-Net is developed for the task of semantic segmentation; when a neural network is fed images as inputs, we can choose to classify objects either generally or by instances. 

![image](https://github.com/aliamrod/PyTorch-and-Monai-for-AI-Healthcare-Imaging/assets/62684338/82fe8a8b-0c2a-4faf-bca9-d1cdd7d5454d)


We can consequently predict what object is in the image (image classification), where all the objects are located (image localization/semantic segmentation), or where
individual objects are located (object detection/instance segmentation). _Figure 02_ below demonstates differences between these computer vision tasks. To simplify 
the matter, we only consider classification for only one class and one label (binary classification). 


![image](https://github.com/aliamrod/PyTorch-and-Monai-for-AI-Healthcare-Imaging/assets/62684338/b685165b-7d3f-46ad-999b-cd47b2e8fc6a)


In the classificaton task, we output a vector of size _k_, where _k_ is the number of classes. In detection tasks, we need to output the vector `x, y, height, width, class`, which define bounding boxes. However, in segmentation tasks, we need to output an image with the same dimension as the original input. This represents quite an engineering challenge-- "Now, how can a neural network extract relevant features from the input image, and then project them into segmentation masks?"

### Encoder-Decoder
The reason why encoder-decoders are relevant is due to the notion that they produce outputs similar to what we desire: output(s) that have the same dimension as the input. Can we apply the concept of encoder-decoder to image segmentation? We can surely generate a one-dimensional binary mask and train the network utilizing cross-entropy loss. In this case, cross-entropy is used to measure the difference between two probability distributions. It is used as a similarity metric to tell how close one distribution of random events are to another, and is used for both classification (in the general sense) as well as segmentation.

Our network consists of two parts: the **encoder** which extracts relevant features from images, and the **decoder** part which takes the extracted features and reconstructs a segmentation mask. 


![image](https://github.com/aliamrod/PyTorch-and-Monai-for-AI-Healthcare-Imaging/assets/62684338/1747c48f-3140-4e86-b98a-ddbbdb1302a5)

<sub>Fig. 03. An encoder-decoder network for image segmentation.</sub>

In the encoder part, I used convolutional layers, followed by `ReLu` and `MaxPool` as the feature extractors. In the decoder part, I transposed convolution to increase the size of the feature map and decrease the number of channels. I utilized padding to maintain the size of the feature maps the same after convolution operations. One thing you may notice is that unlike classification networks, this network does not have a fully connected/linear layer. This is an instance of a **fully convolutional network** (FCN). FCN has been shown to work well on segmentatoin tasks, starting with Shelhamer _et al._ paper "Fully Convolutional Networks for Semantic Segmentation". 



### Software Installation
```
Python
VS Code
3D Slicer
ITK-SNAP
```


### Finding Dataset(s)
For generalizable 3D semantic segmentation medical data, 'medicaldecathlon.com' was consulted. 
```
https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2
```




### Weighted Cross Entropy


### Training


### Testing the Model
To test the model, I have provided the Jupyter notebook testing.ipynb file that contains the different codes that needed to accomplish this portion of the analysis. You will find the part to plot the training/testing graphs about the loss and the dice coefficient and of course you will find the the part to show the results of one of the test data to see the output of your model.

<img width="554" alt="graphs" src="https://github.com/aliamrod/PyTorch-and-Monai-for-AI-Healthcare-Imaging/assets/62684338/89a0a6a8-d6bb-4cee-a0fd-0aa2ac7e610b">





### References

[1] "U-Net: Convolutional Networks for Biomedical Image Segmentation". _International Conference on Medical image computing and computer-assisted intervention._

[2] "Understanding U-Net: U-Net has become the go-to method for image segmentation. But how did it come to be?"

[3] "Fully convolutional networks for semantic segmentation." _Proceedings of the IEEE conference on computer vision and pattern recognition._ 

[4] "V-net: Fully convolutional neural networks for volumetric medical image segmentation." _2016 Fourth Internation Conference on 3D Vision (3DV)._

[5] "Understanding Latent Space in Machine Learning". _Towards Data Science._

[6] "Confounders mediate AI prediction of demographics in medical imaging". 

[7] "Robust and data-efficient generalization of self-supevised machine learning for diagnostic imaging".

[8] "Medical imaging data science competitions should report dataset demographics and evaluate for bias".

[9]

[10]

[11]

[12]

[13]

[14]

[15]

