#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 16:14:58 2019

@author: techvamp
"""

import os
import keras
from Myscripts.NN.Conv2.mynet2 import Mynet2
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image 
import cv2
from keras.preprocessing import image

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
        'Dataset/validation',
        target_size=(100, 100),
        batch_size=20,
        class_mode='categorical')


train_generator = train_datagen.flow_from_directory(
        'Dataset/train',
        target_size=(100, 100),
        save_to_dir = 'Dataset/prep',
        save_prefix= 'train',
        batch_size=20,
        class_mode='categorical')


import numpy
model = Mynet2.build(100,100,3,2)
model.summary()

opti = keras.optimizers.Adam()

model.compile(optimizer = opti, loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit_generator(train_generator, steps_per_epoch = 20, epochs = 15, validation_data = validation_generator, validation_steps = 60)

model.save('Model/trained_model.h5')


