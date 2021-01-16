# Object_Detection_using_Mobilenet_and_Yolov3

## Object detection using deep learning includes Single Shot Detectors and Yolov3

When it comes to deep learning-based object detection there are three primary object detection methods that will likely encounter:

*Faster R-CNNs
*You Only Look Once (YOLO)
*Single Shot Detectors (SSDs)

## STEPS TO BE COVERED:
1. OpenCV’s dnn  module to load a pre-trained object detection network.
2. This will enable us to pass input images or video or webcam real time video through the network and obtain the output bounding box (x, y)-coordinates of each object in the image.

# Object detection using MobileNets
MobileNet is an object detector released in 2017 as an efficient CNN architecture designed for mobile and embedded vision application. This architecture uses proven depth-wise separable convolutions to build lightweight deep neural networks.
It differ from traditional CNNs through the usage of depthwise separable convolution.
The general idea behind depthwise separable convolution is to split convolution into two stages:
* A 3×3 depthwise convolution.
* Followed by a 1×1 pointwise convolution.
* This allows to actually reduce the number of parameters in the network.

# Import all libraries
Having imported all needed libraries, the next step is to write a simple Python script that helps us load images or convert real-time video frames

# Downloading the model file 
* Download the MobileNetV3 pre-trained model to your machine
* Move it to the project folder.
* Create a main.ipynb python script to run the real-time program.

This configuration file defines the model architecture and params.
Initialize model prediction by passing in the config path of the model
Pre-process the image.
Assign a target label to the object in the image.
Predicts the probability of the target label to each frame in the image.

# Putting it all together
The video stream and the writer in place, the next step is to keep the video stream live and perform real-time object detection by looping through the frames catpured from the video stream. As long as this keeps running, we can visually see the object detection result by displaying it on our screen.



