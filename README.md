# **Traffic Sign Recognition** 

## Project Writeup
### By:   Omer Waseem
### Date: 2017-08-21

---

[//]: # (Image References)

[image1]: ./writeup_images/sample_image_09.jpeg "Sample image (no passing)"
[image2]: ./writeup_images/hist.jpeg "Histogram of training data"
[image3]: ./writeup_images/before_hist_eq.jpeg "Before histogram equalization"
[image4]: ./writeup_images/after_hist_eq.jpeg "After histogram equalization"
[image5]: ./new_traffic_signs/001_27.jpeg "Downloaded image: pedestrians"
[image6]: ./new_traffic_signs/002_25.jpeg "Downloaded image: road work"
[image7]: ./new_traffic_signs/003_18.jpeg "Downloaded image: general caution"
[image8]: ./new_traffic_signs/004_11.jpeg "Downloaded image: right of way at next intersection"
[image9]: ./new_traffic_signs/005_13.jpeg "Downloaded image: yield"

## Description
#### This project uses a CNN with 7 weighted layers (4 convolutional and 3 fully connected) to classify the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) with **99.5%** validation accuracy and **97.9%** test accuracy.

---
  
### Files Submitted

#### 1. [Project Writeup (this file)](https://github.com/omerwase/SDC_P2_Traffic_Sign_Classifier/blob/master/README.md)
#### 2. [Traffic_Sign_Classifier.ipynb Notebook](https://github.com/omerwase/SDC_P2_Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)
#### 3. [IPython Notebook Report](https://github.com/omerwase/SDC_P2_Traffic_Sign_Classifier/blob/master/report.html)
#### 4. [New German Traffic Sign Images](https://github.com/omerwase/SDC_P2_Traffic_Sign_Classifier/tree/master/new_traffic_signs/)
  
### Data Set Summary & Exploration

#### 1. Dataset Summary

The following stats about the dataset were calculated using numpy methods and attributes:

* The size of training set is **34799** images
* The size of the validation set is **4410** images
* The size of test set is **12630** images
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

#### 2. Exploratory Visualization

Below is an example image from the dataset corresponding to a **No Passing** sign (label 9):  
![alt text][image1]

The historgram below shows the number of images for each class in the training dataset:  
![alt text][image2]

There is a large discrepancy between certian classes. For example label 0 has 180 examples (lowest) and label 2 has 2010 (highest). One possible enhancement to the training data would be to gather more images of traffic signs with few examples. **Note:** additional data was not generated for this project. Validation accuracy of **99.5%** and test accuracy of **97.9%** was achieved using only the dataset pictured above.

### Design and Test a Model Architecture

#### 1. Preprocessing

All images have two preprocessing methods applied to them:  
* i) Histogram equalization: to increase the contrast in dark images. The example below is of a yield sign, before and after histogram equalization.  
![alt text][image3]![alt text][image4]
* ii) Pixel scaling: all pixels are scale between -0.5 to 0.5 for numerical stability in the network.
  
#### 2. Model Architecture

The final model consisted of the following layers:

| Layer         		    |     Description	        				            	| 
|:---------------------:|:---------------------------------------------:| 
| Input         		    | 32x32x3 RGB image   				            			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					        |							                        					|
| Dropout				        |							                        					|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 24x24x64 	|
| RELU					        |							                        					|
| Dropout				        |							                        					|
| Max pooling	      	  | 2x2 stride,  outputs 12x12x64				        	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 8x8x128   	|
| RELU					        |							                        					|
| Dropout				        |							                        					|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 4x4x256  	|
| RELU					        |							                        					|
| Dropout				        |							                        					|
| Flatten				        |	from 4x4x256 to 4096				         					|
| Fully connected		    | input 4096, outputs 1024			                |
| Fully connected		    | input 1024, outputs 256		  	                |
| Fully connected		    | input 256, outputs 43 (logits)                |
  
#### 3. Model Training
  
The logits from the model above and true one-hot-encoded labels were used to calculate the softmax cross entropy of the network's predictions. The loss was determined by calculating the mean of the cross entropy, which was minimized through the Adam optimizer.
  
Network hyperparameters were tuned to achieve better accuracy. The best result was obtained with: 
* 500 epochs
* 128 batch size
* 0.0001 learning rate
* 0.5 dropout
  
#### 4. Solution Approach
  
The final model results are:
* validation set accuracy of 99.5%
* test set accuracy of 97.9%

To start, LeNet was used to classify the traffic signs which resulted in accuracies around 90%. At this point the only preprocessing applied was pixel scaling. The network was then modify with additional 5x5 convolution layers. Two convolution layers were placed back-to-back (inspired by the VGGNet architecture). This increased the validation accuracy of the network to 95%. Using histogram equalization (in preprocessing) increased this accuracy to 97%. Dropout was added after each convolution layer to avoid overfitting when training for large epochs. Finally the network was trained over 500 epochs which produced the final validation accuracy of 99.5%
  
### Test a Model on New Images

#### 1. Acquiring New Images

Below are the five downloaded images used to test network predictions:
![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8] ![alt text][image9]

All of the images were cropped and resized to 32x32, so they can be fed directly into the network. These images are fairly clean, so it was expected that the network would predict them all accurately. It is worth noting that the first image of the pedestrian is flipped horizontally, when compared to images of the same class in the training dataset. As discussed below, this was the only image the model got wrong.

#### 2. Performance on New Images
Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Below are the model's predictions of the images above (respectively):

| Image			                        | Prediction	                        	  			| 
|:---------------------------------:|:---------------------------------------------:| 
| Pedestrians      	               	| Right-of-way at next intersection        			| 
| Road work     			              | Road work 										                |
| General caution					          | General caution											          |
| Right-of-way at next intersection | Right-of-way at next intersection		         	|
| Yield	      	                  	| Yield      						          	            |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This does not compare well to the test accuracy of 97.9%, the model was expected to have 100% accuracy. The one image the model got wrong was flipped horizontally compared to images of the same class in the training dataset. This is likely the reason for the wrong prediction. One possible way to mitigate these kinds of errors is to train the model on flipped images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

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
