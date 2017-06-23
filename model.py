import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda , Cropping2D,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split

import sklearn
from sklearn.utils import shuffle


samples = []
with open('../../sim-data7/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
               # print( batch_sample[2])
                center_image = cv2.imread( batch_sample[0])
                center_image = cv2.cvtColor(center_image,cv2.COLOR_RGB2YUV)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle * -1)
                
                center_image = cv2.imread( batch_sample[1])
                center_image = cv2.cvtColor(center_image,cv2.COLOR_RGB2YUV)
                center_angle = float(center_angle + 0.2)
                images.append(center_image)
                angles.append(center_angle)
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle * -1)
                
                
                center_image = cv2.imread( batch_sample[2])
                center_image = cv2.cvtColor(center_image,cv2.COLOR_RGB2YUV)
                center_angle = float(center_angle - 0.2)
                images.append(center_image)
                angles.append(center_angle)
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle * -1)
                

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)



# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=10)
validation_generator = generator(validation_samples, batch_size=10)

ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
model.add(Lambda(lambda x : x /255 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24,5,5,subsample = (2,2),activation = "relu"))
model.add(Convolution2D(36,5,5,subsample = (2,2),activation = "relu"))
model.add(Convolution2D(48,5,5,subsample = (2,2),activation = "relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3, activation = "relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)
model.save('model25.h5');
print('saved')