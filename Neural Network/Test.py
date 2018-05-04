import tensorflow as tf
import Train
import numpy as np
from ModelBuilder import read_model
import keras
import ModelBuilder
from Utils import connection_available
from keras.utils import np_utils
from LoadData import GetData
from KFoldCrossValidation import CrossValidate

# defining the folders path train and test
TRAINING_DATASET_FOLDER_NAME = '3_preprocessed_1_dataset train'
TEST_DATASET_FOLDER_NAME = '3_preprocessed_2_dataset test'

folders_at_the_same_time = 60
batch_size = 128            # in each iteration, we consider 128 training examples at once
num_epochs = 180            # we iterate 200 times over the entire training set

X_train, Y_train, _ = (GetData(TRAINING_DATASET_FOLDER_NAME, limit_value=folders_at_the_same_time))
Y_train = np_utils.to_categorical(Y_train, 2)
X_train = X_train.astype('float32')
X_train /= np.max(X_train)

width = 80
height = 80
depth = 2
# num_test = X_test.shape[0] #num test images
num_classes = 2         # there are 2 image classes

# load the model
a = read_model("models/model1.txt")
modelObject = ModelBuilder.ModelBuilder(a, (height, width, depth))
model = modelObject.model

print(model.summary())

model.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
              optimizer='adam',                 # using the Adam optimiser
              metrics=['accuracy'])             # reporting the accuracy

count = 0

# PREVIOUS TRAINING METHOD
model.fit(X_train, Y_train,   # Train the model using the training set...li
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.2, callbacks=[keras.callbacks.LambdaCallback(
                            on_epoch_begin=lambda batch, logs: inc())])

def inc():
    global count
    count += 1
