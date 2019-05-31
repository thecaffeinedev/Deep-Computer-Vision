# Deep Learning For Image captioning
Deep CNN-LSTM for Generating Image Descriptions 

**Key words**: Image captioning, image description generator, explain image, merge model, deep learning, long-short term memory, recurrent neural network, convolutional neural network, word by word, word embeding, bleu score.

#### Abstract
Image captioning is a very interesting problem in machine learning. With the development of deep neural network, deep learning approach is the state of the art of this problem. The main mission of image captioning is to automatically generate an image's description, which requires our understanding about content of images. In the past, there are some end-to-end models which were introduced such as: GoogleNIC ([show and tell](https://arxiv.org/pdf/1411.4555.pdf)), MontrealNIC ([show attend and tell](https://arxiv.org/pdf/1502.03044.pdf)), [LRCN](https://arxiv.org/pdf/1411.4389.pdf), [mRNN](https://arxiv.org/pdf/1410.1090.pdf), they are called inject-model with idea is give image feature throught RNN. In 2017, Marc Tanti, et al. introduce their [paper](https://arxiv.org/pdf/1708.02043.pdf) **What is the Role of Recurrent Neural Networks (RNNs) in an Image Caption Generator?** with merge-model. The main idea of this model is separate CNN and RNN, with only merge their ouput at the end and predicted by softmax layer. Base on it, we develop our model to generate image caption. 

### I. Main Idea:
* Combine ConvNet with LSTM
* Deep ConvNet as image encoder
* Language LSTM as text encoder
* Fully connected layer as decoder
* End-to-end model I -> S
* Maximize P(S|I)

### II. Dataset: 
[Flickr 8k](https://forms.illinois.edu/sec/1713398), train/val/test 6:1:1.

### IV. Implement code:
1. Load images and extract feature: [kaggle-kernel](https://www.kaggle.com/damminhtien/development-model-resnet50)
2. Load text data: [kaggle-kernel](https://www.kaggle.com/damminhtien/text-data-exploxe)
3. Develop model and training: [kaggle-kernel](https://www.kaggle.com/damminhtien/visualization-development-model-resnet50)
4. Evaluation model: [kaggle-kernel](https://www.kaggle.com/damminhtien/evaluate-model)
5. Generator caption for new images: [kaggle-kernel](https://www.kaggle.com/damminhtien/generation-caption-for-new-image)

### V. Tuning hyperparameters:
> ### Encoder ConvNet:
* VGG16
* **Resnet50**
* Densenet121
* Inceptionv3

> ### Optimizer
* **Adam**: 
* Nadam: 
* RMSprop: 
* Sgd:

### VI. Evaluation and result:
> **We use BLEU-score which is evaluate metric:**
+ BLEU-1: 0.542805
+ BLEU-2: 0.301714
+ BLEU-3: 0.207351
+ BLEU-4: 0.095704

