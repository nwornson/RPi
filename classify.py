from utilities import filter_data, image_gather, vgg16_proc_image, proc_image
import pandas as pd
from PIL import Image
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array

# l and h don't matter here?  I think VGG16 preproc will make them 224x224
h=224
l=224

# our test image
im_path = 'test_cropped.jpg'

# process the image
imData = proc_image(im_path,l,h)
## reshape to add extra dimension
imData = imData.reshape((1, imData.shape[0], imData.shape[1], imData.shape[2]))

# load model
model = VGG16()
prob = model.predict(imData)

from keras.applications.vgg16 import decode_predictions

label = decode_predictions(prob)
## unsure here
label1 = label[0][0]

print('%s (%.2f%%)' % (label1[1], label1[2]*100))