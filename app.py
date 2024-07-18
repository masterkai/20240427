from keras.applications import VGG16

model = VGG16(weights="imagenet")
model.summary()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras.utils as image_utils