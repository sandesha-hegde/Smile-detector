#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 11:38:11 2019

@author: techvamp
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 11:16:24 2019

@author: techvamp
"""
import os
import os.path
import numpy as np
import urllib
import glob
import cv2
import keras
import tensorflow as tf
import tensorflow.keras
import pathlib
from PIL import Image 
import keras.callbacks


from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.callbacks import Callback
#image argumentation or preprocessing

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'Dataset/train',
        target_size=(100, 100),
        color_mode = 'rgb',
        batch_size=20,
        class_mode='categorical')


validation_generator = test_datagen.flow_from_directory(
        'Dataset/validation',
        target_size=(100, 100),
        color_mode = 'rgb',
        batch_size=20,
        class_mode='categorical')

classifier = Sequential()

#step 1-Convolution operation
classifier.add(Convolution2D(32, (3, 3), input_shape = (100, 100, 3), activation = 'relu'))

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
classifier.add(Dense(output_dim = 2, activation='sigmoid'))

#compiling CNN
opti = keras.optimizers.adam(lr = 0.001)

classifier.compile(optimizer = opti, loss = 'categorical_crossentropy', metrics = ['accuracy'])
import os


classifier.fit_generator(
        train_generator,
        steps_per_epoch=20,
        epochs=15,
        verbose = 1,
        validation_data=validation_generator,
        validation_steps=60)

classifier.save('Model/keras_fit3_img_prep1.h5')

file = '/home/techvamp/Documents/Project/Smile-Detector/Dataset/validation/natural/1a.jpg'

from keras.preprocessing import image

image = cv2.imread(file)
cv2.imshow('',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
image = cv2.resize(image, (100, 100))
image = np.array(image, np.float32) / 255.0


input_tenser = np.expand_dims(image, axis=0)

what = classifier.predict(input_tenser)[0]