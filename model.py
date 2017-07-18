import pandas as pd
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda , Cropping2D,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from utils import batch_generator
from keras.callbacks import ModelCheckpoint
import os

import sklearn
from sklearn.utils import shuffle




def load_data(data_dir):
    #Load training data and split it into training and validation set

    data_df = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'))

    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_valid, y_train, y_valid




#CNN Model
model = Sequential()
model.add(Lambda(lambda x : x /127.5 - 1.,input_shape=(66,200,3)))

model.add(Convolution2D(24,5,5,activation = "relu", subsample=(2, 2)))
model.add(Convolution2D(36,5,5,activation = "relu", subsample=(2, 2)))
model.add(Convolution2D(48,5,5,activation = "relu", subsample=(2, 2)))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3, activation = "relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100,activation = "relu"))
model.add(Dense(50,activation = "relu"))
model.add(Dense(10,activation = "relu"))
model.add(Dense(1))


data_dir = 'D:/Study/Udacity/Self Driving Car/Term 1/project 3/data/'
X_train, X_valid, y_train, y_valid = load_data(data_dir)
batch_size = 40

checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only='true',
                                 mode='auto')


model.compile(loss='mse', optimizer='adam')
model.fit_generator(batch_generator(data_dir, X_train, y_train, batch_size, True), samples_per_epoch= 20000, validation_data=batch_generator(data_dir, X_valid, y_valid, batch_size, False), nb_val_samples=len(X_valid), nb_epoch=10,callbacks = [checkpoint],verbose = 1)
model.save('model51.h5')
print('saved')