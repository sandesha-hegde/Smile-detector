#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:56:01 2019

@author: techvamp
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

class Mynet2:
    
    def build(height, width, depth, classes):
        classifier = Sequential()
        #step 1-Convolution operation
        classifier.add(Convolution2D(32, (3, 3), input_shape = (height, width, depth), activation = 'relu'))
        
        #step 2 -pooling layer
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        
        #Adding second convlolutional layer
        classifier.add(Convolution2D(64, (3, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        
        #step 3-Flattening layer
        classifier.add(Flatten())
        
        #step 4-Fully connected layers
        classifier.add(Dense(output_dim = 128, activation = 'relu'))
        
        #step5-output layer
        classifier.add(Dense(output_dim = classes, activation='sigmoid'))
        
        return classifier
    