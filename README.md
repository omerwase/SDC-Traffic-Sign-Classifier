# **Traffic Sign Recognition** 


[//]: # (Image References)

[image1]: ./writeup_images/sample_image_09.jpeg "Sample image (no passing)"
[image2]: ./writeup_images/hist.jpeg "Histogram of training data"
[image3]: ./writeup_images/before_hist_eq.jpeg "Before histogram equalization"
[image4]: ./writeup_images/after_hist_eq.jpeg "After histogram equalization"
[image5]: ./new_traffic_signs/001_27.jpg "Downloaded image: pedestrians"
[image6]: ./new_traffic_signs/002_25.jpg "Downloaded image: road work"
[image7]: ./new_traffic_signs/003_18.jpg "Downloaded image: general caution"
[image8]: ./new_traffic_signs/004_11.jpg "Downloaded image: right of way at next intersection"
[image9]: ./new_traffic_signs/005_13.png "Downloaded image: yield"

#### This project uses a CNN with 7 weighted layers (4 convolutional and 3 fully connected) to classify the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) with **99.5%** validation accuracy and **97.9%** test accuracy.

---

### Files Submitted

#### 1. [Project Writeup (this file)](https://github.com/omerwase/SDC_P2_Traffic_Sign_Classifier/blob/master/README.md)
#### 2. [Traffic_Sign_Classifier.ipynb Notebook](https://github.com/omerwase/SDC_P2_Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)
#### 3. [IPython Notebook Report](https://github.com/omerwase/SDC_P2_Traffic_Sign_Classifier/blob/master/report.html)
#### 4. [New German Traffic Sign Images](https://github.com/omerwase/SDC_P2_Traffic_Sign_Classifier/tree/master/new_traffic_signs/)
  
---

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
  
Below are the model's predictions of the images above (respectively):
  
| Image			                        | Prediction	                        	  			| 
|:---------------------------------:|:---------------------------------------------:| 
| Pedestrians      	               	| Right-of-way at next intersection        			| 
| Road work     			              | Road work 										                |
| General caution					          | General caution											          |
| Right-of-way at next intersection | Right-of-way at next intersection		         	|
| Yield	      	                  	| Yield      						          	            |
  
The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This does not compare well to the test accuracy of 97.9%, the model was expected to have 100% accuracy. The one image the model got wrong was flipped horizontally compared to images of the same class in the training dataset. This is likely the reason for the wrong prediction. One possible way to mitigate these kinds of errors is to train the model on flipped images.
  
#### 3. Model Certainty - Softmax Probabilities

The code for calculating the top 5 softmax probabilities is located in the 22nd cell of the IPython notebook.  
  
The top 5 softmax probabilities for each image are listed below (respectively). The model is very certian in all its predictions, which seems odd. The reason for this is currently unknown, and any feedback on why this might be the case would be highly appreciated. For the first image (which was incorrectly predicted) the 2nd highest softmax probability is for the correct label. However that 2nd probability of 0.0006 is much lower than the 1st and incorrect probability of 0.9993.
  
**Image 1 (Pedestrians):**  

| Probability         	|     Prediction	        				            	| 
|:---------------------:|:---------------------------------------------:| 
| 0.9993         		  	| Right-of-way at the next intersection   		 	| 
| 6.467e-4     			  	| Pedestrians       								        		|
| 7.211e-6					    | Dangerous curve to the left			            	|
| 1.801e-6	      			| Double curve          			 	          			|
| 1.108e-6				      | Beware of ice/snow                           	|
  
**Image 2 (Road work):** 

| Probability         	|     Prediction	        				            	| 
|:---------------------:|:---------------------------------------------:| 
| 0.9998         		  	| Road work   								                	| 
| 2.064e-4     			  	| Beware of ice/snow								        		|
| 9.754e-6					    | Traffic signals									            	|
| 3.153e-6	      			| Speed limit (120km/h)				 	          			|
| 2.126e-6				      | Right-of-way at the next intersection       	|
  
**Image 3 (General caution):** 

| Probability         	|     Prediction	        				            	| 
|:---------------------:|:---------------------------------------------:| 
| 1.000         		  	| General caution							                	| 
| 2.599e-8     			  	| Pedestrians       								        		|
| 3.241e-13					    | Right-of-way at the next intersection        	|
| 1.473e-14	      			| Dangerous curve to the left 	          			|
| 6.301e-16				      | No passing for vehicles over 3.5 metric tons 	|
  
**Image 4 (Right-of-way at next intersection):**  

| Probability         	|     Prediction	        				            	| 
|:---------------------:|:---------------------------------------------:| 
| 1.000         		  	| Right-of-way at the next intersection        	| 
| 1.130e-9     			  	| Beware of ice/snow								        		|
| 1.055e-10					    | Pedestrians   									            	|
| 4.935e-12	      			| Double curve            		 	          			|
| 3.377e-12				      | Vehicles over 3.5 metric tons prohibited     	|
  
**Image 5 (Yield):**  

| Probability         	|     Prediction	        				            	| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000         		  	| Yield       								                	| 
| 0.0     			      	| Speed limit (20km/h)							        		|
| 0.0					          | Speed limit (30km/h)						            	|
| 0.0	      		      	| Speed limit (50km/h)				 	          			|
| 0.0   				        | Speed limit (60km/h)                         	|
