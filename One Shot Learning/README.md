# One Shot Learning 

### Problem Statement

* [Omniglot](https://github.com/brendenlake/omniglot) the “transpose” of MNIST, with 1623 character classes, each with 20 examples. 

* Use background set of 30 alphabets for training and evaluate on set of 20 alphabets. Refer to this script for sampling setup.

* Report one-shot classification (20-way) results using a meta learning approach like [MAML](https://arxiv.org/pdf/1703.03400.pdf).

#### 1. Introduction
Deep Convolutional Neural Networks have become the state of the art methods for image classification tasks. However, one of the biggest limitations is they require a lots of labelled data. In many applications, collecting this much data is sometimes not feasible. One Shot Learning aims to solve this problem.

#### 2. Dataset

The Omniglot dataset which is a collection of 1623 hand drawn characters from 50 different alphabets. For every character there are just 20 examples, each drawn by a different person. Each image is a gray scale image of resolution 105x105.

Before I continue, I would like to clarify the difference between a character and an alphabet. In case of English the set A to Z is called as the alphabet while each of the letter A, B, etc. is called a character. Thus we say that the English alphabet contains 26 characters (or letters).

So I hope this clarifies the point when I say 1623 characters spanning over 50 different alphabets.

Thus we have 1623 different classes(each character can be treated as a separate class) and for each class we have only 20 images. Clearly, if we try to solve this problem using the traditional image classification method then definitely we won’t be able to build a good generalized model. And with such less number of images available for each class, the model will easily overfit.

* You can download the dataset by cloning this [GitHub](https://github.com/brendenlake/omniglot) repository. The folder named “Python” contains two zip files: images_background.zip and images_evaluation.zip. Just unzip these two files.

* images_background folder contains characters from 30 alphabets and will be used to train the model, 
* while images_evaluation folder contains characters from the other 20 alphabets which we will use to test our system.

#### 3. Model Architecture and Training

![Model](https://github.com/TheCaffeineDev/Fellowship.ai-Challenges/blob/master/One%20Shot%20Learning/model.png)

The model was compiled using the adam optimizer and binary cross entropy loss function as shown below. Learning rate was kept low as it was found that with high learning rate, the model took a lot of time to converge. However these parameters can well be tuned further to improve the present settings.

The model was trained for 20000 iterations with batch size of 32.

After every 200 iterations, model validation was done using 20-way one shot learning and the accuracy was calculated over 250 trials.

#### 4. Conclusion

This is just a first cut solution and many of the hyper parameters can be tuned in order to avoid over fitting. Also more rigorous testing can be done by increasing the value of ’N’ in N-way testing and by increasing the number of trials.

* Note: The model was trained on Google Colab.
