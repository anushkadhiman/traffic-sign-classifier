import cv2 
import numpy as np
import matplotlib.image as mimg
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras import backend as K

import os
import glob

K.set_image_dim_ordering('th')

path = 'C:/Users/AD/Downloads/Images/'

data = []
labels = []
img_paths = glob.glob(os.path.join(path, '*/*.ppm'))
print(len(img_paths))
print(type(img_paths))

for img in img_paths:
    imag = mimg.imread(img)
    im1=cv2.resize(imag,(64,64))
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    label = int(img.split('\\')[-2])
    data.append(im1)
    labels.append(label)

data = np.array(data, dtype='float32')  
labels = np.array(labels, dtype='float32')  

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1, stratify=labels)  # splitting data into train-test

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 64, 64)
X_test = X_test.reshape(X_test.shape[0], 1, 64, 64)

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

print(X_train.shape)
print(y_train.shape)

# create model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(1, 64, 64), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=10, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
print("Loss: " + str(scores[0]))
print("Accuracy: " + str(scores[1]))
