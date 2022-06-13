import cv2
import tensorflow as tf
import numpy as np

cnn = tf.keras.models.load_model('model-a.h5')

def predict_cls(img_path):
    img = cv2.imread(img_path)
    img1 = cv2.resize(img,(145,145))
    
    shp = (1,145,145,3)
    arr_t = np.ones(shp)
    for i in range(0,1):
        for j in range(0, 145):
            for k in range(0,145):
                for l in range(0,3):
                    arr_t[i,j,k,l] = img1[j,k,l]
    
    pred = cnn.predict(arr_t)
    
    print (pred)
    return pred


import os 
no_yawn = []
yawn = []
opn = []
close = []

files = os.listdir('archive/train/no_yawn/')
for f in files:
    f = 'archive/train/no_yawn/' + f
    no_yawn.append(predict_cls(f))
    
files = os.listdir('archive/train/yawn/')
for f in files:
    f = 'archive/train/yawn/' + f
    yawn.append(predict_cls(f))
    
files = os.listdir('archive/train/open/')
for f in files:
    f = 'archive/train/open/' + f
    opn.append(predict_cls(f))
    
files = os.listdir('archive/train/closed/')
for f in files:
    f = 'archive/train/closed/' + f
    close.append(predict_cls(f))
    
count_no_ywn = [0,0,0,0]
count_ywn = [0,0,0,0]
count_opn = [0,0,0,0]
count_close = [0,0,0,0]

for pred_ar in no_yawn:
    high = -255
    prediction = -1
    for i in range(0,4):
        if(pred_ar[0,i] > high):
            high = pred_ar[0,i]
            prediction = i
    count_no_ywn[prediction]+=1;

for pred_ar in yawn:
    high = -255
    prediction = -1
    for i in range(0,4):
        if(pred_ar[0,i] > high):
            high = pred_ar[0,i]
            prediction = i
    count_ywn[prediction]+=1;

for pred_ar in opn:
    high = -255
    prediction = -1
    for i in range(0,4):
        if(pred_ar[0,i] > high):
            high = pred_ar[0,i]
            prediction = i
    count_opn[prediction]+=1;

for pred_ar in close:
    high = -255
    prediction = -1
    for i in range(0,4):
        if(pred_ar[0,i] > high):
            high = pred_ar[0,i]
            prediction = i
    count_close[prediction]+=1;



