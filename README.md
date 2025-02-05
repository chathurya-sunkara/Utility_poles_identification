# Utility_poles_identification
This project uses a modified YOLOv9 model to identify utility poles from Google Street View images. It includes data preprocessing, model modifications, and training pipelines. The goal is to automate utility pole detection for efficient infrastructure mapping and management.
# To provides detailed information about the GPUs installed on your system
! nvidia-smi
# To know current working directory
import os

HOME = os.getcwd()

print(HOME)

Knowing the current working directory in a deep learning model setup is essential for managing file paths for loading datasets, saving models, logs, and other outputs in the correct location, ensuring portability and organization.
# Setting the environment
!git clone https://github.com/SkalskiP/yolov9.git

%cd yolov9

!pip install -r requirements.txt -q

In this project YOLOv9 was used so  these commands set up the environment by downloading the necessary code, navigating to the project directory, and installing the required dependencies, ensuring the deep learning model can be trained or used for inference.
# Install Roboflow python package
!pip install -q roboflow

The datset is present in Roboflow. Roboflow is used because It simplifies dataset handling and model deployment in computer vision tasks. When working with YOLO or similar models, Roboflow helps streamline the process of downloading annotated datasets, automatically formatting them for your model, and deploying the model for inference.
# Downloading weights
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt

!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt

!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt

!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt

!ls -la {HOME}/weights

These commands download pre-trained YOLOv9 model weights (yolov9-c.pt, yolov9-e.pt, gelan-c.pt, gelan-e.pt) from GitHub into the weights directory in the current working directory, and !ls -la {HOME}/weights lists the contents of that directory to verify the downloaded files.
# Downloading Dataset
from roboflow import Roboflow

rf = Roboflow(api_key="KHWoNndSaVRmM6vndb5K")

project = rf.workspace("chathurya").project("utility-poles-identification-iqbcw")

version = project.version(2)

dataset = version.download("yolov9")

These lines initialize the Roboflow API using an API key, access the "utility-poles-identification-iqbcw" project in the "chathurya" workspace, retrieve version 2 of the dataset, and download it in the YOLOv9 format for use in the utility pole identification model.

The dataset consists of Google Street View Images and prepared by using various Pre-processing ,augmentation techniques and annotated using bounding boxes.

# Tuning the hyperparameters and Training
%cd {HOME}/yolov9

!python train.py \

--batch 16 --epochs 20 --img 640 --device 0 --min-items 0 --close-mosaic 15 \

--data {dataset.location}/data.yaml \

--weights {HOME}/weights/gelan-c.pt \

--cfg models/detect/gelan-c.yaml \

--hyp hyp.scratch-high.yaml

These lines change the directory to the yolov9 folder, then run the train.py script to train a YOLOv9 model using the specified batch size, number of epochs, image size, device, and other hyperparameters, while loading the dataset and pre-trained weights for initialization.
# To check files
!ls {HOME}/yolov9/runs/train/exp/

This command lists the contents of the runs/train/exp/ directory in the yolov9 folder to inspect the files generated during the training process, such as model weights, logs, and training results.
# Displaying Loss Graphs
from IPython.display import Image

Image(filename=f"{HOME}/yolov9/runs/train/exp/results.png", width=1000)

This code imports the Image class from the IPython.display module and displays the training results image (results.png) generated during training from the specified path, with a width of 1000 pixels.
# Displaying Confusion matrix
from IPython.display import Image

Image(filename=f"{HOME}/yolov9/runs/train/exp/confusion_matrix.png", width=900)

This code imports the Image class from the IPython.display module and displays the confusion matrix image (confusion_matrix.png) generated during the training process from the specified path, with a width of 900 pixels.
# Evaluate model
%cd {HOME}/yolov9

!python val.py \

--img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 \

--data {dataset.location}/data.yaml \

--weights {HOME}/yolov9/runs/train/exp/weights/best.pt 

These lines change the directory to the yolov9 folder, then run the val.py script to evaluate the trained YOLOv9 model on a dataset, using the specified image size, batch size, confidence threshold, IOU threshold, device, and model weights to validate its performance.
# Detection
!python detect.py \

--img 1280 --conf 0.1 --device 0 \

--weights {HOME}/yolov9/runs/train/exp/weights/best.pt \

--source {dataset.location}/test/images

These lines run the detect.py script to perform object detection on the test images from the dataset using the specified model weights, image size, confidence threshold, and device, generating predictions for the given test images.
