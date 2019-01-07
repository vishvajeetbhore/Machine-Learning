#from __future__ import print_function
import os, sys, json, time

import os, sys, json
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.applications.vgg16 import VGG16
from keras import backend as K
from PIL import Image

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image


import LoadCaltech101

# Size of the image
image_size = (224, 224, 3)
# Load the data
((X_train, Y_train), (X_test, Y_test)) = LoadCaltech101.load_data(
datadir="data",
h5file="output/Caltech101.h5",
target_size=image_size,
train_test_ratio=0.5)
N_train = X_train.shape[0]
N_test = X_test.shape[0]
nb_classes = Y_train.shape[1]

print("Training data:", X_train.shape, Y_train.shape)
print("Test data:", X_test.shape, Y_test.shape)

#-----

vgg16 = VGG16(weights='imagenet', include_top=True, input_shape=image_size)

get_activity = K.function(
    [vgg16.layers[0].input,K.learning_phase()],
    [vgg16.layers[-2].output]
)

features_train = get_activity([X_train,0])[0]
features_test = get_activity([X_test])[0]

np.save('output/feature_train', features_train)
np.save('output/features_test', features_test)
np.save('output/Y_train', Y_train)
np.save('output/Y_test', Y_test)

print("Training data:", X_train.shape, Y_train.shape)
print("Test data:", X_test.shape, Y_test.shape)

