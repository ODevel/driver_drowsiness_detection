How to use this repository:
--------------------------
1. model-x.py: where x can be a, b and c depending upon the model. Use this
file to train the appropriate model and generate .h5 ckpt file.
2. restore-a,b.py: Use this file get a live demo of what the DL model is
capable of and how it can be used to detect the drowsiness.
3. predict_img.py: If not using live video capture, use this file to predict
drowziness from a given image.

Dependencies:
-------------
1. Python 3.7
2. Tensor Flow
3. Open CV
4. Numpy

References:
-----------
Use following kaggle dataset to download images while training:
https://www.kaggle.com/code/adinishad/driver-drowsiness-using-keras/data
