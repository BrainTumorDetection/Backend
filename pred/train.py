# OpenCV library module 
import cv2
# OS module for basic file/directory/path manipulation 
import os 
import tensorflow as tf 
from tensorflow import keras 
from keras.utils import normalize
# Pillow library for handling images
from PIL import Image
# NumPy for numerical op esp arrays 
import numpy as np
# SkLearn library to import fun that can split data into train and split
from sklearn.model_selection import train_test_split

# how does this work lmao - shouldn't it be ../datasets/
image_directory='datasets/'

# create list of yes/no images 
no_tumor_images=os.listdir(image_directory+'no/')
yes_tumor_images=os.listdir(image_directory+'yes/')

# list that contained processed input data that model will learn from
dataset=[]
# contain target OUTPUTS for each corresponding entry in datasets (0 for no tumor, 1 for tumor) 
label=[]
# model later learns by comparing predictions against actual levels

# process images in 'no' category 
for i, image_name in enumerate(no_tumor_images):
    # check if file is JPG image, if so read image and convert to RGB array
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory+'no/'+image_name)
        image = Image.fromarray(image, 'RGB')
        # resize image to standardize input size for model 
        image = image.resize((64, 64), Image.Resampling.LANCZOS)  # Specify the resampling filter
        # store processed images 
        dataset.append(np.array(image))
        label.append(0)

# process images in 'yes' category 
for i, image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory+'yes/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64, 64), Image.Resampling.LANCZOS)  # Specify the resampling filter
        dataset.append(np.array(image))
        label.append(1)

# convert to array to use sklearn 
dataset=np.array(dataset)
label=np.array(label)

#print(dataset)
#print(label)

# allocate 80% of images for training, 20% for testing, same split every time run code
dataset_train, dataset_test, dataset_train, label_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# adjust data so pixel intensity on same scale 
x_train=normalize(dataset_train, axis=1)
x_test=normalize(dataset_test, axis=1)

