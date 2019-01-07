from __future__ import print_function
import os, sys, json, time
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam

#from keras.applications.vgg16 import VGG16
#from keras import backend as K

import LoadCaltech101

# Size  of the image
image_size = (224, 224, 3)

#----------------------load input-------------------

X_train = np.load('output/feature_train.npy')
Y_train = np.load('output/Y_train.npy')

X_test = np.load('output/features_test.npy')
Y_test = np.load('output/Y_test.npy')

nb_classes = 3
input_dim = 4096

#model definition


model = Sequential()
model.add(Dense(3, input_shape=(4096,), activation='softmax'))

optimizer = Adam()

model.compile(
    loss= 'categorical_crossentropy',
    optimizer = optimizer,
    metrics = ['accuracy']
)

print(model.summary())

model.fit(X_train, Y_train, batch_size=1, epochs=20, validation_split=0.1)

score = model.evaluate(X_test,Y_test, verbose=0)

print('\nTest loss:', score[0])
print('Test acc:', score[1])



