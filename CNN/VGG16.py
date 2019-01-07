from __future__ import print_function
import time
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from PIL import Image

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image



from keras.applications.vgg16 import VGG16
vgg16 = VGG16(weights='imagenet', include_top=True)
#print(vgg16)

# Load the cat image and resize it to 224x224
img = image.load_img('image_0001.jpg', target_size=(224, 224))
# Convert to Numpy array (224, 224, 3)
x = image.img_to_array(img)
# Add an empty dimension in front to obtain a tensor (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)
# Perform mean removal as in the original VGG16 network
x = preprocess_input(x)
# Make the prediction using VGG16
output = vgg16.predict(x)

#print(output)
plt.figure()
plt.plot(output[0,:],'o')
plt.show()

top5 = decode_predictions(output)

for idx,label,proba in top5[0]:
    print(label, ';' , proba)

