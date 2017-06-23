**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/normal.png 
[image3]: ./examples/rec1.png "Recovery Image"
[image4]: ./examples/rec2.png "Recovery Image"
[image5]: ./examples/rec3.png "Recovery Image"
[image6]: ./examples/normal.png "Normal Image"
[image7]: ./examples/fliped.jpg "Flipped Image"



#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md  summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The Notebook file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 64 (line 72 - 86) 

The model includes RELU layers to introduce nonlinearity (line 76), and the data is normalized in the model using a Keras lambda layer (line 73). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (lines 81). 

The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the Nvidia's model I thought this model might be appropriate because it has been well tested and used.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding dropout layer

#### 2. Final Model Architecture

The final model architecture (lines 76-86) consisted of a convolution neural network with the following layers and layer sizes 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						| 
| Lambda Layer     		| Preprocessing layer , normazlises the image 	|
| Cropping				| Removes the unwanted regions 					|
| Convolution 3x3		| Input 24x5x5,sumbsample=(2,2),Activation=Relu |
| Convolution 3x3		| Input 36x5x5,sumbsample=(2,2),Activation=Relu |
| Convolution 3x3		| Input 48x5x5,sumbsample=(2,2),Activation=Relu |
| Convolution 3x3		| Input 64x5x5,sumbsample=(2,2),Activation=Relu |
| Convolution 3x3		| Input 64x5x5,sumbsample=(2,2),Activation=Relu |
| Dropout				| keep probablity = 0.5							|
| Flatten				|												|
| Fully connected		| input 100 									|
| Fully connected		| input 50  									|
| Fully connected		| input 10  									|
| Fully connected		| input 1    									|
|						|												|
 

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself if it drifts away. These images show what a recovery looks like.

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also flipped images and angles thinking that this would give more data points. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]



After the collection process, I had 49,000 number of data points. I then preprocessed this data by normalizing the pixle and intensites and zero centring it.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10. I used an adam optimizer so that manually training the learning rate wasn't necessary.