# Object_Detection_using_Mobilenet_and_Yolov3

## Object detection using deep learning includes Single Shot Detectors and Yolov3

When it comes to deep learning-based object detection there are three primary object detection methods that will likely encounter:

* Faster R-CNNs
* You Only Look Once (YOLO)
* Single Shot Detectors (SSDs)

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


# Object detection using YOLOV3

It forwards the whole image only once through the network.
YOLOv3 gives faster than realtime results on a M40, TitanX or 1080 Ti GPUs.
YOLO makes detections at three different scales (three different sizes at three different places in the network) 
* First, it divides the image into a 13×13 grid of cells. 
* The size of these 169 cells vary depending on the size of the input. 
* For a 416×416 input size that we used in our experiments, the cell size was 32×32. Each cell is then responsible for predicting a number of boxes in the image.
* For each bounding box, the network also predicts the confidence that the bounding box actually encloses an object, and the probability of the enclosed object being a particular class.
* Most of these bounding boxes are eliminated because their confidence is low or because they are enclosing the same object as another bounding box with very high confidence score. This technique is called non-maximum suppression.

 # Import all libraries
Having imported all needed libraries, the next step is to write a simple Python script that helps us load images or convert real-time video frames

 # Download the models
 * yolov3.weights file (containing the pre-trained network’s weights)
 * yolov3.cfg file (containing the network configuration)
 * coco.names file which contains the 80 different class names used in the COCO dataset.
 
 # Initialize the parameters
* confThreshold = 0.5  #Confidence threshold
* nmsThreshold = 0.4   #Non-maximum suppression threshold

# Load the model and classes
* yolov3.weights : The pre-trained weights.
* yolov3.cfg : The configuration file.
* coco.names contains all the objects for which the model was trained.

# Putting it all together
The video stream and the writer in place, the next step is to keep the video stream live and perform real-time object detection by looping through the frames catpured from the video stream. As long as this keeps running, we can visually see the object detection result by displaying it on our screen.

## CONCLUSION

SSD is one of the object detection algorithm that forwards the image once though a deep learning network, but YOLOv3 is much faster than SSD while achieving very comparable accuracy. 
The art advances in object detection model YOLO from version one to version three, as well as the advances made in image classification model MobileNet from version one to
version three. Both the models, namely YOLO and MobileNet are compared with other detection frameworks and network architectures. Progress in these
architectures will enable the AI dream to come true.
