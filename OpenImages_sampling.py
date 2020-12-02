from utilities import filter_data, image_gather, vgg16_proc_image, proc_image
import pandas as pd
from PIL import Image
import numpy as np
from keras.applications.vgg16 import VGG16

# import the data
df = pd.read_csv('validation-annotations-human-imagelabels-boxable.csv')
class_desc = pd.read_csv('class-descriptions-boxable.csv',header = None, names = ['LabelName','Label'])

# merge datasets
full_df = pd.merge(df,class_desc,how = 'left',on = 'LabelName')

# specify height, length, and relevant labels
#  a sample of 'other' is included by default
h=224
l=224
flist = ['Bicycle','Car','Dog','Person']


imageID_df = filter_data(full_df,flist)

ids = list(imageID_df['ImageID'])

imData = image_gather(ids,h,l)
print(imData.shape)

# view an image
idx = np.random.randint(len(imData),size=1)
testIm = imData[idx,:,:,:]

# check label
imageID = ids[int(idx)]
print(imageID_df[imageID_df['ImageID']==imageID])

from keras.preprocessing.image import load_img, img_to_array
#from_direc = load_img('validation/e86b1d0bf7235885.jpg',target_size=(224,224))
#from_direc=img_to_array(from_direc)
model = VGG16()

# preprocess_input() subtracts the mean, mine only scales
#from keras.applications.vgg16 import preprocess_input
#from_direc = preprocess_input(from_direc)
#from_direc = from_direc.reshape((1, from_direc.shape[0], from_direc.shape[1], from_direc.shape[2]))
path = 'validation/' + imageID + '.jpg'
print(type(path))
imData = proc_image(path,l,h)
# since single image, need to reshape
imData = imData.reshape((1, imData.shape[0], imData.shape[1], imData.shape[2]))

prob = model.predict(imData)

from keras.applications.vgg16 import decode_predictions

label = decode_predictions(prob)
## unsure here
label1 = label[0][0]

print(imageID_df[imageID_df['ImageID']==imageID])
print('%s (%.2f%%)' % (label1[1], label1[2]*100))



#im = Image.fromarray((testIm).astype(np.uint8)) # unscale the image
#im.show()