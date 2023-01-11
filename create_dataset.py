#----------------------------------BIBLIOTEKI-------------------------------------------------
import cv2               
import numpy as np        
import os                
from random import shuffle
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy
import math
import multiprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import InputLayer, add, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib


#-----------------------------------LOSOWE NAMNAZANIE OBRAZOW----------------------------------
train_datagen = ImageDataGenerator(
        rotation_range=40,
        rescale=1/1,
        shear_range=0.2,
        zoom_range=0.2,
	data_format='channels_last',
        horizontal_flip=False)

valid_datagen = ImageDataGenerator(
        rotation_range=40,
        rescale=1/1,
        shear_range=0.2,
        zoom_range=0.2,
	data_format='channels_last',
        horizontal_flip=False)

#----------------------------------------IMAGE DATA GENERATOR-----------------------------------
pathTrain='/home/ewa-laptop/Desktop/sztuczna inteligencja/train'
pathTest='/home/ewa-laptop/Desktop/sztuczna inteligencja/valid'

train_batches = train_datagen.flow_from_directory(pathTrain, target_size = (28,28),classes=['fist','hand','ok','onefinger','turnleft','turnright'], batch_size=600, color_mode = 'grayscale',)
valid_batches = valid_datagen.flow_from_directory(pathTest, target_size = (28,28),classes=['fist','hand','ok','onefinger','turnleft','turnright'], batch_size=240,color_mode = 'grayscale',)

img,labels = next(train_batches)

#--------------------------------------MODEL----------------------------------------------------
model=tf.keras.Sequential() 
model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(28, 28,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy')

model.fit_generator(train_batches, steps_per_epoch=100,validation_data=valid_batches,validation_steps=40, epochs=20,verbose=2)

model.save('model.h5')

print("Saved model to disk")

