#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 22:16:54 2019

@author: techvamp
"""

from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import cv2
import keras
import matplotlib
import numpy as np
import imutils
import argparse

#get the  dataset,video,cascade from the path mentioned i arguements
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--cascade', required = True, help = 'path for cascade classifier')
ap.add_argument('-m', '--models', required = True, help = 'path for saved model')
ap.add_argument('-v', '--video', required = True, help = 'video file')

args = vars(ap.parse_args())

cascade = args['cascade']

#Load haar cascade for face detection
detector = cv2.CascadeClassifier(cascade)

# Load the model for smile classifier
classifier = keras.models.load_model(args['models'])

#if a video path was not supplied, grab the refrences to the webcam
if not args.get('video', False):
    print('[INFO] starting video capture...')
    camera = cv2.VideoCapture(0)

# otherwise, load the video
else:

    video = cv2.VideoCapture(args['video'])


while True:
    ret, frame = video.read()
   # fram = cv2.imread(frame)
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clone = frame.copy()
    #detect the faces in the input frame
    rect = detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (fX, fY, fW, fH) in rect:
        
        roi = gray[fW:fX+fW, fH:fY+fH]
        
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        #pass the detected image to the classifier to get the prediction
        (notsmiling, smiling) = classifier.predict(roi)[0]
     # label the generated output
        label = 'smiling' if smiling > notsmiling else 'notsmiling'
        # Drow rectangle when smiles and display text on the box
        if label == 'smiling' :
            cv2.putText(clone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            cv2.rectangle(clone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)
       # else:
          #  cv2.putText(clone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            #cv2.rectangle(clone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
            
     # display the video   
    cv2.imshow('face', clone)

        
    if cv2.waitKey(80) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()