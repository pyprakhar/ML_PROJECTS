# Object Detection using YOLOv8 on Custom Dataset

This repository contains Python code for Object Detection using YOLOv8 on a Custom Dataset. The YOLOv8 model is a state-of-the-art deep learning model for real-time object detection.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Custom Dataset](#custom-dataset)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Object Detection](#object-detection)
- [Results and Insights](#results-and-insights)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview
Object detection is a computer vision task that involves detecting and localizing objects in images or videos. YOLOv8 is a powerful object detection model that can identify multiple objects in real-time with high accuracy.

## Dependencies
To run the code in this repository, you need the following Python libraries:
- PyTorch
- OpenCV
- Numpy

You can install these dependencies using the following command:
```
!pip install ultralytics
```

## Custom Dataset
For this project, we use a custom dataset containing images with annotated objects that we want to detect. The dataset should be structured in a way that YOLOv8 can understand, i.e., in the format of image files along with corresponding annotation files (e.g., YOLO format .txt files).

## Data Preparation
- Organize the custom dataset into train, validation, and test sets.
- Prepare annotation files for each image in the YOLO format, specifying the object's class and bounding box coordinates.
- custom data set made on robo flow https://universe.roboflow.com/galgotias/car-detection-model-uieuk

## Model Training
- Load the YOLOv8 model and pre-trained weights.
- Fine-tune the model on the custom dataset using transfer learning.
- Adjust hyperparameters and train the model on the training set.

## Model Evaluation
- Evaluate the trained model's performance on the validation set using relevant metrics such as mean average precision (mAP) and accuracy.

## Object Detection
- Use the trained YOLOv8 model to perform real-time object detection on new images or videos.
- Visualize the detected objects with bounding boxes and class labels.

## Results and Insights
Analyze the results of object detection, including any challenges faced and potential improvements to the model's performance.

## Usage
- Clone this repository:
```
git clone https://github.com/pyprakhar/object-detection-yolov8.git
```
- Navigate to the project directory:
```
cd object-detection-yolov8
```
- Prepare the custom dataset and annotations in the required format.
- Run the Python script for training the model:
```
python train_yolov8.py
```
- Run the Python script for real-time object detection:
```
python detect_objects.py
```

## Contributing
Contributions to this repository are welcome! If you have any suggestions, improvements, or bug fixes, feel free to create a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
