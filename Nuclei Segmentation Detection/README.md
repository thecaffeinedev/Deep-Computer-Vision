# Nuclei Segmentation By Mask RCNN 

This directory contains solution for nuclei instance segmentation using Mask R-CNN, including image pre-processing, Mask R-CNN with training augmentation, test stage ensemble and post-processing.

The code for Mask R-CNN model is adapted from [MatterPort implementation.](https://github.com/matterport/Mask_RCNN)

***Dataset*** : [Kaggle 2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018) 

***[Link to my Colab Notebook](https://colab.research.google.com/drive/1rrH_ExLT5DbFC5CIneNN5k0ivECDKn34)***

## Overview

Run the Notebook On Google Colab

1. Goto Colab 
2. Goto File-->Upload Notebook . 
3. Goto menu Runtime-->Change runtime and select HardWare accelerator GPU 
4. Execute all cells

### Training 
The model is trained on NVIDIA K80 GPU.

### Dependencies 
* Python version 3
* skimage 
* Tensorflow 
* Keras 
* Pandas 

### Data Description & Challenge

The dataset is challenging because of high volume and dimensionality. Our data is divided into a training set (665 images, each containing between 4 to 384 masks for distinct nuclei) and test set (65 images). The images vary in size (total pixels) and were collected from many different cell types under a variety of imaging conditions (magnification, modality, etc). To achieve success, we will have to work with all the given data to develop a robust method for cell nucleus identification.


### COMPUTER VISION TECHNIQUES
Object detection, and specifically segmentation of cell nuclei, within the field of computer vision technique is a well studied concept. Despite object detection and image segmentation being the focus of many research groups, this topic in computer vision still has much room for growth and improvement. 

A relatively recent, and significant, advancement in image segmentation was the integration of convolutional neural networks into computer vision algorithms. CNNs are particularly adept at image classification due to their ability to identify patterns in images. 
The depth of convolutional neural networks can be increased to then merge patterns into higher level feature detection, and eventually robust classification. 

In the application of cell nuclei segmentation and masking, image classification alone isn’t particularly helpful. The next step is to label any and all objects in an image as nucleus or not nucleus (background). This step is accomplished by employing a regional convolutional neural network (R-CNN). The Mask R-CNN model requires a validation set to calculate loss during training. The validation set has been sourced from the training set by performing an 80:20 split on the data. 80% of the data, 536 images, are used for training and the remaining 20%, 134 images, are used for validation.
