import tensorflow as tf
import Train
import numpy as np
from ModelBuilder import read_model
import ModelBuilder
from Utils import connection_available
from keras.utils import np_utils
from LoadData import GetData
from ModelBuilder import model_array_builder
from KFoldCrossValidation import CrossValidate
import os

import keras
from Utils import current_datetime
from ModelBuilder import read_model
from ModelBuilder import ModelBuilder


# ====================CONFIGURING GPU ========================================
config = tf.ConfigProto()
config.gpu_options.allow_growth = False

# ============================================================================

enable_telegram_bot = True if connection_available() else False
# chat_id = 125016709               # this is my private chat id
# chat_id = "@gdptensorboard"       # this is the name of the public channel
chat_id = -1001223624517            # this is for the private channel

# defining the folders path train and test
TRAINING_DATASET_FOLDER_NAME = '3_preprocessed_1_dataset train'
TEST_DATASET_FOLDER_NAME = '3_preprocessed_2_dataset test'

epochs_with_same_data = 10
folders_at_the_same_time = 25
validation_folders = 1
validate_every = 2

batch_size = 128            # in each iteration, we consider 128 training examples at once
num_epochs = 1            # we iterate 200 times over the entire training set

height = 80
width = 80
depth = 2
num_classes = 2

# weight of the classes, when an error occour on class 0 -> false positive.
class_weight = {0: 1, 1: 1}

timestamp = current_datetime()

# load data
(X_validation, y_validation, validation_folders_list) = GetData(TRAINING_DATASET_FOLDER_NAME, limit_value=validation_folders)
X_validation = X_validation.astype('float32')
X_validation /= np.max(X_validation)    # Normalise data to [0, 1] range
Y_validation = np_utils.to_categorical(y_validation, num_classes)   # One-hot encode the labels


# load models
m1 = read_model("models/model01.txt")
modelObject1 = ModelBuilder(m1, (80, 80, 2))
model1 = modelObject1.model
model1.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])


Train.SingletonTrain().Train(
                model1, training_dataset_folder_name=TRAINING_DATASET_FOLDER_NAME, epochs=20, batch_size=batch_size,
                training_folders_count=folders_at_the_same_time, validation_x=X_validation, validation_y=Y_validation,
                to_avoid=validation_folders_list, validate_every=validate_every, subfolder_name=timestamp,
                enable_telegram_bot=(chat_id != "undefined"), save_model="model01", validation_treshold=0.015
)


# =============================

m2 = read_model("models/model02.txt")
modelObject2 = ModelBuilder(m2, (80, 80, 2))
model2 = modelObject2.model
model2.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])

Train.SingletonTrain().Train(
                model2, training_dataset_folder_name=TRAINING_DATASET_FOLDER_NAME, epochs=150, batch_size=batch_size,
                training_folders_count=folders_at_the_same_time, validation_x=X_validation, validation_y=Y_validation,
                to_avoid=validation_folders_list, validate_every=validate_every, subfolder_name=timestamp,
                enable_telegram_bot=(chat_id != "undefined"), save_model="model02", validation_treshold=0.015
)
