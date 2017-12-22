# Semantic-Segmentation_SelfDrivingCar_Term3_P2
This repo contains the submissions and related material for Udacity "Self Driving Car" Nano degree program's Term 3 - Project 2, "Semantic-Segmentation"

## Introduction

The main goal of this project is to perform semantic segmentation on an image to highlight the road, or the drivable path for self driving car.

# [Click here for small presentation video on output images](https://www.youtube.com/watch?v=Vkf4cMnO7ig&t=31s)

- From Test image

![alt text](https://github.com/sulabhmatele/Semantic-Segmentation_SelfDrivingCar_Term3_P2/blob/master/images/umm_000008.png)

To

![alt text](https://github.com/sulabhmatele/Semantic-Segmentation_SelfDrivingCar_Term3_P2/blob/master/runs_epochs50/1513740327.5142293/umm_000008.png)

- From Test image

![alt text](https://github.com/sulabhmatele/Semantic-Segmentation_SelfDrivingCar_Term3_P2/blob/master/images/um_000017.png)

To

![alt text](https://github.com/sulabhmatele/Semantic-Segmentation_SelfDrivingCar_Term3_P2/blob/master/runs_epochs50/1513740327.5142293/um_000017.png)

Project labels the pixels of a road in images using a Fully Convolutional Network (FCN).
## Datasets

For the project we use
 - Pretrained VGG16 model Link [https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip]
 - Kitti Road dataset Link [http://www.cvlibs.net/datasets/kitti/eval_road.php]
 - Architechtural Ref. Link [https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf]

## Fully Convolutional Netwrok

To save the spacial information for the image and locate the object on a part of image, we need a technique which preserves this information.

![alt text](https://github.com/sulabhmatele/Semantic-Segmentation_SelfDrivingCar_Term3_P2/blob/master/images/FullFCN_Architecture.png)

Fully convolutional neural network is a technieque which preserves the spacial information, this information is generally lost in general convolutional neural network.

Fully convolution network implementation can be viewed in following steps:

1. Converting fully connected layer of general convolutional neural network to 1x1 convolution layer.

2. Implementing a deconvolution on output from 1x1 layer.

3. Implementing skip layer connection for better learning.

![alt text](https://github.com/sulabhmatele/Semantic-Segmentation_SelfDrivingCar_Term3_P2/blob/master/images/SkipConnections.png)

The whole technique also referred as Encoder decoder pattern, where the pretrained model is considered as Encoder and the deconvolution model is referred as Decoder.

![alt text](https://github.com/sulabhmatele/Semantic-Segmentation_SelfDrivingCar_Term3_P2/blob/master/images/FCN_EnDeView.png)

## Steps of implementation

- Load the pretrained VGG16 model.

- Define the layers for deconvolution, where we feed output of pretrained model to 1x1 convolution and then implement skip connections and transpose convolution.
Deconvolution also referred as upsampling process since we upsample the 1x1 image to original size image.

- Implement optimizer for effective learning. I used AdamOptimizer.

- Train and run the model.

- Save the output images.

### Hyperparameters

```
Epochs            : 50
Batch size        : 8
Learning rate     : 0.0001
Keep Probability  : 0.6
```

### Setup

##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

##### Run
Run the following command to run the project:
```
python main.py
```

#### References

+ Udacity lesson screen shots for architechture and images.
+ https://discussions.udacity.com/t/gpu-running-out-of-memory-for-semantic-segmentation-project/396747/6


