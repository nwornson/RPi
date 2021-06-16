# Useful functions for sampling from a large data set of labelled images (i.e ImageNet)

# Dependencies
import pandas as pd
from PIL import Image
import numpy as np

# pre processing function
def proc_image(path,x,y):
    image = Image.open(path).resize((x,y))
    
    data = np.asarray(image)
    
    data = data.astype('float32')

    # centering
    mean = data.mean(axis=1)
    centered = data - mean[:, None]

    # scaling
    #data /= 255
    
    return centered

# process image specifically for VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

def vgg16_proc_image(path):
    from_direc = load_img(path,target_size=(224,224))
    data = img_to_array(from_direc)
    # preprocess_input centers data
    data = preprocess_input(path)

    return data

# taking a random sample
def sample_column(data,label,val):
    sample = data.loc[data['Label'] == label]
    out = sample.sample(n=val)
    return out

def filter_data(df,labels):
    
    # filter the data based on the chosen labels
    Olabels=df[~df['Label'].isin(labels)]['Label'].unique()
    df['Label'] = df['Label'].replace(Olabels,'Other')
    
    
    filtered_df = df[df['Label'].isin(labels)]
    
    df = df[['ImageID','Label']]
  
    # find minimum class frequency
    
    min_val = filtered_df['Label'].value_counts().min()
    
    labels.append('Other')
    
    # define a new data frame
    out_df = pd.DataFrame({
        'ImageID':[],
        'Label':[]
    })
    
    for label in labels:
        fildf = sample_column(df,label,min_val)
        out_df = pd.concat([out_df,fildf])
    
    
    
    return out_df


# assemple image tensor (rgb)
def image_gather(im_ids,h,l):
    # get shape
    total = len(im_ids)
    data_array = np.zeros(shape=(total,h,l,3))
    grey_counter = 0
    for i in range(0,total):
        path = 'validation/' + im_ids[i] + '.jpg'
        data = proc_image(path,l,h)
        # skip greyscale images
        if data.shape != (h,l,3):
           grey_counter += 1
        else:
            idx = i - grey_counter
            data_array[idx,:,:,:] = data
        
    idxs = np.arange(total - grey_counter,total)
    data_array = np.delete(data_array,idxs,axis=0)
        
    return data_array