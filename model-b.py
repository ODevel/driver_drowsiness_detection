# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('archive/train/',
                                                 target_size = (145, 145),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

# Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('archive/train',
                                            target_size = (145, 145),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# Initialising the CNN
cnn = tf.keras.models.Sequential()

cnn.add(Conv2D(256, (3, 3), activation="relu", input_shape=[145,145,3]))
cnn.add(MaxPooling2D(2, 2))
cnn.add(Conv2D(128, (3, 3), activation="relu"))
cnn.add(MaxPooling2D(2, 2))
cnn.add(Conv2D(128, (3, 3), activation="sigmoid"))
cnn.add(MaxPooling2D(2, 2))
cnn.add(Conv2D(128, (3, 3), activation="sigmoid"))
cnn.add(MaxPooling2D(2, 2))
cnn.add(Flatten())
cnn.add(Dropout(0.5))
cnn.add(Dense(64, activation="relu"))
cnn.add(Dense(4, activation="softmax"))


# Compiling the CNN
cnn.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam") 

# Training 
history = cnn.fit(training_set, epochs=20, validation_data=test_set, shuffle=True, validation_steps=len(test_set))

cnn.save('model-b.h5')

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))

import matplotlib.pyplot as plt
plt.plot(epochs, accuracy, "b", label="trainning accuracy")
plt.plot(epochs, val_accuracy, "r", label="validation accuracy")
plt.legend()
plt.show()

plt.plot(epochs, loss, "b", label="trainning loss")
plt.plot(epochs, val_loss, "r", label="validation loss")
plt.legend()
plt.show()

#labels_new = ["yawn", "no_yawn", "Closed", "Open"]

