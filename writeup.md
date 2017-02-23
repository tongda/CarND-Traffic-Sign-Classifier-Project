# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[train-label-distribution]: ./writeup/train-label-distribution.png "Train Label Visualization"
[valid-label-distribution]: ./writeup/valid-label-distribution.png "Valid Label Visualization"
[preprocess]: ./writeup/preprocess.png "preprocess Comparing"
[normalization-formula]: ./writeup/normalization_formula.svg "Normalization Formula"
[sign1]: ./downloaded/1.jpg "Traffic Sign 1"
[sign2]: ./downloaded/2.jpg "Traffic Sign 2"
[sign3]: ./downloaded/3.jpg "Traffic Sign 3"
[sign4]: ./downloaded/4.jpg "Traffic Sign 4"
[sign5]: ./downloaded/5.jpg "Traffic Sign 5"
[sign6]: ./downloaded/6.jpg "Traffic Sign 6"
[sign7]: ./downloaded/7.jpg "Traffic Sign 7"
[sign8]: ./downloaded/8.jpg "Traffic Sign 8"
[sign9]: ./downloaded/9.jpg "Traffic Sign 9"
[sign10]: ./downloaded/10.jpg "Traffic Sign 10"
[sign1-r]: ./downloaded/1-r.jpg "Traffic Sign 1 Resized"
[sign2-r]: ./downloaded/2-r.jpg "Traffic Sign 2 Resized"
[sign3-r]: ./downloaded/3-r.jpg "Traffic Sign 3 Resized"
[sign4-r]: ./downloaded/4-r.jpg "Traffic Sign 4 Resized"
[sign5-r]: ./downloaded/5-r.jpg "Traffic Sign 5 Resized"
[sign6-r]: ./downloaded/6-r.jpg "Traffic Sign 6 Resized"
[sign7-r]: ./downloaded/7-r.jpg "Traffic Sign 7 Resized"
[sign8-r]: ./downloaded/8-r.jpg "Traffic Sign 8 Resized"
[sign9-r]: ./downloaded/9-r.jpg "Traffic Sign 9 Resized"
[sign10-r]: ./downloaded/10-r.jpg "Traffic Sign 10 Resized"

## Rubric Points

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? - **34799**
* The size of test set is ? - **12630**
* The shape of a traffic sign image is ? - **(32, 32, 3)**
* The number of unique classes/labels in the data set is ? - **43**

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set for training. It is a bar chart showing how the data is distributed among different traffic sign.

![alt text][train-label-distribution]

Here is an exploratory visualization of the data set for validation.

![alt text][valid-label-distribution]

As the images show, we can tell that the data is biased, which means data of some categories are significantly higher than others.

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided not to convert the images to grayscale because I believe that converting images to grayscale will lose information, which may be hidden in different channel. However, grayscale image may be helpful, so I decided to add the grayscale images into the original images as fourth channel, which is similar in what the paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks" mentioned.

Next, I normalized the image with four channels to (0, 1), using formula (referenced from [Wikipedia](https://en.wikipedia.org/wiki/Normalization_%28image_processing%29)):

![normalization-formula][normalization-formula]

As comparing, I show the original, grayscale and normalized 3-channel here.

![preprocess][preprocess]

From the left and right images, we can tell that after normalization, the color in the image seems more balanced.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I use the default split of train and validation set.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the fifth cell of the ipython notebook.

My final model consisted of the following layers:

| Layer         		| Sublayer  |     Description	        					|
|:---------------------:|:-:|:---------------------------------------------:|
| Input         		| | 32x32x3 RGB image   							|
| Inception (branch 1x1)     	| Convolution 1x1	| 1x1 stride, outputs 32x32x16  |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch 3x3)      | Convolution 1x1	| 1x1 stride, outputs 32x32x16 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|     	| Convolution 3x3 | 1x1 stride, outputs 32x32x32 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|       | Convolution 3x3 | 1x1 stride, outputs 32x32x32 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch 5x5)      | Convolution 1x1	| 1x1 stride, outputs 32x32x16 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|     	| Convolution 5x5 | 1x1 stride, outputs 32x32x32 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch pool)      | Average Pooling 3x3	| 1x1 stride, outputs 32x32x3 |
|     	| Convolution 3x3 | 1x1 stride, outputs 32x32x16 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Concatenation |   | Concatenate all the branches to 32x32x96 |
| Dropout       |   |     |
| Inception (branch 1x1)     	| Convolution 1x1	| 1x1 stride, outputs 32x32x16  |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch 3x3)      | Convolution 1x1	| 1x1 stride, outputs 32x32x16 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|     	| Convolution 3x3 | 1x1 stride, outputs 32x32x32 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|       | Convolution 3x3 | 1x1 stride, outputs 32x32x32 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch 5x5)      | Convolution 1x1	| 1x1 stride, outputs 32x32x16 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|     	| Convolution 5x5 | 1x1 stride, outputs 32x32x32 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch pool)      | Average Pooling 3x3	| 1x1 stride, outputs 32x32x3 |
|     	| Convolution 3x3 | 1x1 stride, outputs 32x32x16 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Concatenation |   | Concatenate all the branches to 32x32x96 |
| Inception (branch 1x1)     	| Convolution 1x1	| 2x2 stride, outputs 16x16x32  |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch 3x3)      | Convolution 1x1	| 1x1 stride, outputs 32x32x32 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|     	| Convolution 3x3 | 1x1 stride, outputs 32x32x64 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|       | Convolution 3x3 | 2x2 stride, outputs 16x16x64 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch 5x5)      | Convolution 1x1	| 1x1 stride, outputs 32x32x32 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|     	| Convolution 5x5 | 2x2 stride, outputs 16x16x64 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch pool)      | Average Pooling 3x3	| 1x1 stride, outputs 32x32x96 |
|     	| Convolution 3x3 | 2x2 stride, outputs 16x16x32 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Concatenation |   | Concatenate all the branches to 16x16x192 |
| Dropout       |   |     |
| Inception (branch 1x1)     	| Convolution 1x1	| 1x1 stride, outputs 16x16x32  |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch 3x3)      | Convolution 1x1	| 1x1 stride, outputs 16x16x32 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|     	| Convolution 3x3 | 1x1 stride, outputs 16x16x64 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|       | Convolution 3x3 | 1x1 stride, outputs 16x16x64 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch 5x5)      | Convolution 1x1	| 1x1 stride, outputs 16x16x32 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|     	| Convolution 5x5 | 1x1 stride, outputs 16x16x64 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch pool)      | Average Pooling 3x3	| 1x1 stride, outputs 16x16x192 |
|     	| Convolution 3x3 | 1x1 stride, outputs 16x16x32 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Concatenation |   | Concatenate all the branches to 16x16x192 |
| Inception (branch 1x1)     	| Convolution 1x1	| 2x2 stride, outputs 8x8x64  |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch 3x3)      | Convolution 1x1	| 1x1 stride, outputs 16x16x64 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|     	| Convolution 3x3 | 1x1 stride, outputs 16x16x128 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|       | Convolution 3x3 | 2x2 stride, outputs 8x8x128 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch 5x5)      | Convolution 1x1	| 1x1 stride, outputs 16x16x64 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|     	| Convolution 5x5 | 2x2 stride, outputs 8x8x128 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch pool)      | Average Pooling 3x3	| 1x1 stride, outputs 16x16x192 |
|     	| Convolution 3x3 | 2x2 stride, outputs 8x8x64 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Concatenation |   | Concatenate all the branches to 8x8x384 |
| Dropout       |   |     |
| Inception (branch 1x1)     	| Convolution 1x1	| 1x1 stride, outputs 8x8x64  |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch 3x3)      | Convolution 1x1	| 1x1 stride, outputs 8x8x64 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|     	| Convolution 3x3 | 1x1 stride, outputs 8x8x128 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|       | Convolution 3x3 | 1x1 stride, outputs 8x8x128 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch 5x5)      | Convolution 1x1	| 1x1 stride, outputs 8x8x64 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|     	| Convolution 5x5 | 1x1 stride, outputs 8x8x128 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch pool)      | Average Pooling 3x3	| 1x1 stride, outputs 8x8x192 |
|     	| Convolution 3x3 | 1x1 stride, outputs 8x8x64 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Concatenation |   | Concatenate all the branches to 8x8x384 |
| Dropout       |   |     |
| Inception (branch 1x1)     	| Convolution 1x1	| 2x2 stride, outputs 4x4x128  |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch 3x3)      | Convolution 1x1	| 1x1 stride, outputs 8x8x128 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|     	| Convolution 3x3 | 1x1 stride, outputs 8x8x128 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|       | Convolution 3x3 | 2x2 stride, outputs 4x4x256 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch 5x5)      | Convolution 1x1	| 1x1 stride, outputs 8x8x128 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|     	| Convolution 5x5 | 2x2 stride, outputs 4x4x256 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch pool)      | Average Pooling 3x3	| 1x1 stride, outputs 8x8x384 |
|     	| Convolution 3x3 | 2x2 stride, outputs 4x4x128 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Concatenation |   | Concatenate all the branches to 4x4x768 |
| Dropout       |   |     |
| Inception (branch 1x1)     	| Convolution 1x1	| 1x1 stride, outputs 4x4x128  |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch 3x3)      | Convolution 1x1	| 1x1 stride, outputs 4x4x128 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|     	| Convolution 3x3 | 1x1 stride, outputs 4x4x256 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|       | Convolution 3x3 | 1x1 stride, outputs 4x4x256 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch 5x5)      | Convolution 1x1	| 1x1 stride, outputs 4x4x128 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|     	| Convolution 5x5 | 1x1 stride, outputs 4x4x256 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch pool)      | Average Pooling 3x3	| 1x1 stride, outputs 4x4x768 |
|     	| Convolution 3x3 | 1x1 stride, outputs 4x4x128 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Concatenation |   | Concatenate all the branches to 4x4x768 |
| Inception (branch 1x1)     	| Convolution 1x1	| 2x2 stride, outputs 2x2x256  |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch 3x3)      | Convolution 1x1	| 1x1 stride, outputs 4x4x256 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|     	| Convolution 3x3 | 1x1 stride, outputs 4x4x256 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|       | Convolution 3x3 | 2x2 stride, outputs 2x2x512 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch 5x5)      | Convolution 1x1	| 1x1 stride, outputs 4x4x256 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
|     	| Convolution 5x5 | 2x2 stride, outputs 2x2x512 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Inception (branch pool)      | Average Pooling 3x3	| 1x1 stride, outputs 4x4x768 |
|     	| Convolution 3x3 | 2x2 stride, outputs 2x2x512 |
|     	| Batch Normalization	| 0.9997 Moving Average Decay  |
|     	| RELU	|   |
| Concatenation |   | Concatenate all the branches to 2x2x1536 |
| Dropout       |   |     |
| Average Pool 2x2  |   | 2x2 stride, outputs 1x1x1536|
| Dropout       |   |     |
| Fully connected   |   | Outputs 43 									|
| Softmax   				|   | -       									|

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the sixth, seventh and eighth cell of the ipython notebook.

In the sixth cell, the code are helpers about training, including creating placeholders, loss operation, optimizer updating batch normalization variables and combining them all into one train operation. Because the labels in dataset are not one-hot encoding, so I have to convert labels into one-hot encoding using `tf.one_hot`. I chose `AdamOptimizer` as optimizer because it can avoid getting stuck in local optimal.

In the seventh cell, the code is about evaluation of model. These code are similar to the LeNet code in the course.

In the eighth cell, the code is about the real work of training.
* The batch size is set to 512. I am training the model on Nvidia Titan X, which can provide great power.
* The epoch number is set to 100. Because the final version of model is a big model, which need more epochs to train.
* The variables `losses`, `validation_accuracies`, `train_accuracies` are used to record the trending of loss and accuracy during training.
* For each epoch, calling `shuffle` to randomize the training dataset.
* Splitting batches and training them are similar to the code in LeNet in the course.
* Using Python `time.time` function to measuring the time used for training and evaluation.
* Every 10 epochs, print the metrics.
* Every 10 epochs, save the model, so that I can choose the best model according to the validation accuracy.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ?
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The first architecture I chose was LeNet with replacing the final fully connected layer to a layer with 43 output neurons. The test accuracy of LeNet is around 90%.

** TODO: add diagram here **

* What were some problems with the initial architecture?

As the paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks" shows, the baseline are all more than 95%, so I decide to look for better architecture.

The first attempt was to add more convolutional layers. I tried 9 convolutional layers and the performance get improved. As the diagram shows below, the training accuracy and validation accuracy are higher than original LeNet model. But the gap between training accuracy and validation is to big. So that's where I came up with regularization.

** TODO: add diagram here **

As what I have learned in [Stanford CS231n](http://cs231n.stanford.edu/), when the ConvNet goes deeper, the classic L2 regularization do not work well. Instead, I decided to use batch normalization and dropout to avoid overfitting. So I got the third model as follows:

** TODO: add diagram here **

I think the capability still has space to improve, so I decided to involve a more complicated model, inception, which is mentioned in the lecture video. Then after many tuning, I got the final model.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

When adjusting my model, I followed some rules:

1. If the model is under fitting, which means the training accuracy is low, then enlarge the capability of the model, by adding more convolution layers, adding more neurons in some layers, reducing pooling layers, or using more complicated model, such as inception.
2. If the model is over fitting, which means the gap between training accuracy and validation accuracy is high, then adding more regularization, by adding more dropout layers, or lower the dropout keeping probability.
3. Another thing worth to be aware of is the weight initialization. Since we use batch normalization, so we just use truncated normal initialization instead Xavier initialization. I set the mean to 0.01 instead of 0 to avoid dead neuron, because I use RELU as activation function. Actually, I tried ELU to replace RELU, but it did not prove to be helpful.

* Which parameters were tuned? How were they adjusted and why?

To be honest, I did not tune hyperparameters very carefully. What I have done is just as follows:

1. Try some different learning rate (0.01, 0.001, 0.0001).
2. Try some different dropout keep probability (0.5, 0.6. 0,7, 0.8).
3. Try some different number of epochs (10, 50, 100, 200).

I know I should try different combination of these parameters, but the time is limited, so I just choose a looks-better set of parameters.

If a well known architecture was chosen:

* What architecture was chosen?

I chose inception architecture.

* Why did you believe it would be relevant to the traffic sign application?

When I was tuning the model, I found that I need to decide what the filter size should be, 5x5, 3x3, or 1x1? Then I recall that inception architecture is to use all of them, concatenate the output of different branch, the model will choose the best after well trained.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The validation accuracy is around 97%, and test accuracy is around 97%, which is much higher than the prior models. Considering that I did not use some complex preprocessing black magic, I am satisfied with this result.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][sign1] ![alt text][sign2] ![alt text][sign3]
![alt text][sign4] ![alt text][sign5] ![alt text][sign6]
![alt text][sign7] ![alt text][sign8] ![alt text][sign9]
![alt text][sign10]

Because the model only consume images with 32x32x3 size, before testing with them, I resized them to 32x32x3 using `PIL.Image.resize`. The resized images are as follows:

![alt text][sign1-r] ![alt text][sign2-r] ![alt text][sign3-r]
![alt text][sign4-r] ![alt text][sign5-r] ![alt text][sign6-r]
![alt text][sign7-r] ![alt text][sign8-r] ![alt text][sign9-r]
![alt text][sign10-r]

There are some challenges to test the model on these images:

1. Some signs are not in the training set, such as No.2, No.3, No.4, and No.6.
2. After resized, the signs are not very clear, such as No.2, No.5, No.8 and No.9.
3. The sign are not segmented as well as training data, such as No.5, No.8, No.9 and No.10.
4. Some of the images has watermark on them, which may affect the accuracy of recognition.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...
