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
