from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils
from LoadData import GetData
from LoadData import ResultPrediction
from keras.models import load_model
from time import time
from keras.callbacks import TensorBoard
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
import math
import time
import datetime
from tensorflow.python.client import device_lib

import tensorflow as tf
#============================================================================
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#============================================================================

#defining the folders path train and test
TRAINING_DATASET_FOLDER_NAME = '1_dataset train'
TEST_DATASET_FOLDER_NAME = '2_dataset test'

batch_size = 32 # in each iteration, we consider 32 training examples at once
num_epochs = 1 # we iterate 200 times over the entire training set
kernel_size = 4 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth_1 = 2 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 4 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.1 # dropout after pooling with probability 0.25
drop_prob_2 = 0.2 # dropout in the FC layer with probability 0.5
hidden_size = 50 # the FC layer will have 512 neurons

(X_train, y_train), (X_test, y_test) = (GetData(TRAINING_DATASET_FOLDER_NAME), GetData(TEST_DATASET_FOLDER_NAME)) # fetch data

print(X_train.shape)

num_train, height, width, depth = X_train.shape
num_test = X_test.shape[0] #num test images
num_classes = 2 # there are 2 image classes

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= np.max(X_train) # Normalise data to [0, 1] range
X_test /= np.max(X_test) # Normalise data to [0, 1] range

Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels


inp = Input(shape=(height,width,depth ))
# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(pool_1)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)

pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
#drop_1 = Dropout(drop_prob_1)(pool_1)
# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(pool_2)
pool_3 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_3)
conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(pool_3)
pool_4 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
#drop_2 = Dropout(drop_prob_1)(pool_2)
# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
flat = Flatten()(pool_4)
hidden = Dense(hidden_size, activation='relu')(flat)
hidden2 = Dense(hidden_size, activation='relu')(hidden)
#drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(hidden2)

model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy


#====== configuring tensorboard ======

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit(X_train, Y_train,   # Train the model using the training set...
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.3, callbacks=[tbCallBack]) # ...holding out 10% of the data for validation
loss = model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!



print(model.metrics_names)
print(loss)
predicted_label = model.predict(X_test)

for i in range(1,2):
    #print(np.squeeze(X_test[i], axis=0).shape)
    #plt.figure(1)
    result = (predicted_label[i])
    real_value = (Y_test[i])
    #plt.subplot(math.ceil(len(X_test/8)), 1, 1)
    plt.imshow(np.squeeze(X_test[i], axis=2), 'gray')
    plt.title(ResultPrediction(result, real_value))
    plt.show()

    #print(result)
    #print(real_value)


#====== save model ======

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

model.save('trained_model/' + st + '.h5')
#math.ceil(len(X_test/100)

