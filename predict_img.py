# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import cv2
import os
import numpy as np

cnn = tf.keras.models.load_model('model-a.h5')

test_dir = 'archive/test/'
labels = ['Closed', 'Open', 'no_yawn', 'yawn']

def verify_img(img_path):
    img = cv2.imread(img_path)
    img1 = cv2.resize(img,(145,145))
    
    shp = (1,145,145,3)
    arr_t = np.ones(shp)
    for i in range(0,1):
        for j in range(0, 145):
            for k in range(0,145):
                for l in range(0,3):
                    arr_t[i,j,k,l] = img1[j,k,l]
    

    result = cnn.predict(arr_t)

    high = -255
    prediction = -1
    for i in range(0,4):
        if(result[0,i] > high):
            high = result[0,i]
            prediction = i

    return labels[prediction]

closed_files = os.listdir('archive/test/Closed/')
open_files = os.listdir('archive/test/Open/')
yawn_files = os.listdir('archive/test/yawn/')
no_yawn_files = os.listdir('archive/test/no_yawn/')

result = verify_img('archive/test/Closed/' + closed_files[10])
print(result)
