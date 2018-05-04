import tensorflow as tf
import Train
import numpy as np
from ModelBuilder import read_model
import ModelBuilder
from Utils import connection_available
from keras.utils import np_utils
from LoadData import GetData
from KFoldCrossValidation import CrossValidate


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

epochs_with_same_data = 3
folders_at_the_same_time = 30
validation_folders = 12
validate_every = 5

batch_size = 128            # in each iteration, we consider 128 training examples at once
num_epochs = 180            # we iterate 200 times over the entire training set

# (X_train, y_train), (X_test, y_test) = (GetData(TRAINING_DATASET_FOLDER_NAME, limit_on_fonders_to_fetch = True, limit_value = 4), GetData(TEST_DATASET_FOLDER_NAME)) # fetch data

height = 80
width = 80
depth = 2
# num_test = X_test.shape[0] #num test images
num_classes = 2         # there are 2 image classes

#weight of the classes, when an error occour on class 0 -> false positive.
class_weight = {0: 1.5, 1: 1}


# inp = Input(shape=(height, width, depth))
#
# # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
# conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
# pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
# drop_1 = Dropout(drop_prob_conv)(pool_1)
# conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
# pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
# drop_2 = Dropout(drop_prob_conv)(pool_2)
# conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_2)
# pool_3 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_3)
# drop_3 = Dropout(drop_prob_conv)(pool_3)
# conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_3)
#
#
# # Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
# flat = Flatten()(conv_4)
#
# hidden = Dense(hidden_size, activation='relu')(flat)
# drop_4 = Dropout(drop_prob_hidden)(hidden)
# hidden2 = Dense(hidden_size, activation='relu')(drop_4)
# drop_5 = Dropout(drop_prob_hidden)(hidden2)
# hidden3 = Dense(hidden_size, activation='relu')(hidden2)
#
# out = Dense(num_classes, activation='softmax')(hidden3)
#
# model = Model(inputs=inp, outputs=out)      # To define a model, just specify its input and output layers

# load the model
a = read_model("models/model1.txt")
modelObject = ModelBuilder.ModelBuilder(a, (height, width, depth))
model = modelObject.model

print(model.summary())

model.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
              optimizer='adam',                 # using the Adam optimiser
              metrics=['accuracy'])             # reporting the accuracy

# configuring training sequence
print("Loading {} validation folders...".format(validation_folders))

(X_validation, y_validation, validation_folders_list) = GetData(TRAINING_DATASET_FOLDER_NAME, limit_value=validation_folders)
X_validation = X_validation.astype('float32')
X_validation /= np.max(X_validation)    # Normalise data to [0, 1] range
Y_validation = np_utils.to_categorical(y_validation, num_classes)   # One-hot encode the labels

# configuring callbacks
"""
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=1, verbose=1, mode='auto')
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=True)
validation_callback = ValidationCallback(X_validation, Y_validation, validate_every=1, chat_id=chat_id)

facesequence = FaceSequence(batch_size, TRAINING_DATASET_FOLDER_NAME,
                            epochs_with_same_data=epochs_with_same_data,
                            folders_at_the_same_time=folders_at_the_same_time,
                            to_avoid=validation_folders_list, enable_telegram_bot=enable_telegram_bot,
                            chat_id=chat_id)
# do the actual training
Train(model, facesequence, num_epochs, chat_id=chat_id, training_callbacks=[validation_callback])
"""

print("Starting training")
history = Train.SingletonTrain().Train(model, training_dataset_folder_name=TRAINING_DATASET_FOLDER_NAME,
                                       epochs=num_epochs, batch_size=batch_size,
                                       epochs_with_same_data=epochs_with_same_data,
                                       training_folders_count=folders_at_the_same_time, validation_x=X_validation,
                                       validation_y=Y_validation, to_avoid=validation_folders_list,
                                       validate_every=validate_every, class_weight=class_weight,
                                       enable_telegram_bot=enable_telegram_bot)

# TO-DO: test the model
#models = [model, model]
#CrossValidate(2, models, TRAINING_DATASET_FOLDER_NAME, batch_size=batch_size, num_epochs=4, folders_at_the_same_time=folders_at_the_same_time, epochs_with_same_data=2, validate_every=2)


# PREVIOUS TRAINING METHOD
# model.fit(X_train, Y_train,   # Train the model using the training set...li
#          batch_size=batch_size, epochs=num_epochs,
#          verbose=1, validation_split=0.3, callbacks=[tbCallBack, earlyStopping, changedata]) # ...holding out 30% of the data for validation
