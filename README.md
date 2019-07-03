# Smile-tracker
Using CNN to detect smile from a pre-recorded video or live video

## Description
Smile detector can be used to detect smile in a images,videos,or live stream from a camera.This project used some deep learning techniques to find a smile in an image or video

## Requirements
- keras
- opencv
- matplotlib
- numpy
- imutils
- argparse

if you find any difficulties in installing use below command.
##### pip install requirement.txt

## Preview
![](bean_smile.gif)

## Procedure
- Before detecting the face, we need model which should detect the smiley face.To train the model run `train_model.py` with your own dataset.
- This project used sequential model to build image classifier.
- For detection run `smile_detector.py` with the cascade,video,model you trained

## Credits
- Adrian Rosebrock creator of PyimageSearch
- keras documentation

## Note

- Those who want the trained model you can contact me through e-mail sandeshangiras@gmail.com
