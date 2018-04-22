import numpy as np
import tensorflow as tf
import keras.callbacks
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

from Utils import current_datetime

import telegram
from ModelBuilder import read_model
import ModelBuilder

from LoadData import GetData
from FaceSequence import FaceSequence
from ValidationCallback import  ValidationCallback


# ====================CONFIGURING GPU ========================================
config = tf.ConfigProto()
config.gpu_options.allow_growth = False

# ============================================================================

enable_telegram_bot = True
# chat_id = 125016709               # this is my private chat id
# chat_id = "@gdptensorboard"       # this is the name of the public channel
chat_id = -1001223624517            # this is for the private channel

# defining the folders path train and test
TRAINING_DATASET_FOLDER_NAME = '3_preprocessed_1_dataset train'
TEST_DATASET_FOLDER_NAME = '3_preprocessed_2_dataset test'

epochs_with_same_data = 3
folders_at_the_same_time = 30
validation_folders = 12

batch_size = 128            # in each iteration, we consider 128 training examples at once
num_epochs = 180            # we iterate 200 times over the entire training set

# (X_train, y_train), (X_test, y_test) = (GetData(TRAINING_DATASET_FOLDER_NAME, limit_on_fonders_to_fetch = True, limit_value = 4), GetData(TEST_DATASET_FOLDER_NAME)) # fetch data
(X_validation, y_validation, validation_folders_list) = GetData(TRAINING_DATASET_FOLDER_NAME, limit_value=validation_folders)
(X_train, y_train, _) = GetData(TRAINING_DATASET_FOLDER_NAME, limit_value=folders_at_the_same_time, to_avoid=validation_folders_list)

print(X_train.shape)
num_train, height, width, depth = X_train.shape
# num_test = X_test.shape[0] #num test images
num_classes = 2 # there are 2 image classes

X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
X_train /= np.max(X_train)              # Normalise data to [0, 1] range
# X_test /= np.max(X_test)              # Normalise data to [0, 1] range
X_validation = X_validation.astype('float32')
# X_test = X_test.astype('float32')
X_validation /= np.max(X_validation)    # Normalise data to [0, 1] range

Y_train = np_utils.to_categorical(y_train, num_classes)             # One-hot encode the labels
# Y_test = np_utils.to_categorical(y_test, num_classes)             # One-hot encode the labels
Y_validation = np_utils.to_categorical(y_validation, num_classes)   # One-hot encode the labels

#load the model
a = read_model("models/model1.txt")
modelObject = ModelBuilder.ModelBuilder(a, (height, width, depth))
model = modelObject.model
print(model.summary())

model.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
              optimizer='adam',                 # using the Adam optimiser
              metrics=['accuracy'])             # reporting the accuracy

# INSTRUCTION TO ABLE THE EARLY STOPPING
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=1, verbose=1, mode='auto')

# ====== configuring tensorboard ======
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=True)

# PREVIOUS TRAINING METHOD
# model.fit(X_train, Y_train,   # Train the model using the training set...li
#          batch_size=batch_size, epochs=num_epochs,
#          verbose=1, validation_split=0.3, callbacks=[tbCallBack, earlyStopping, changedata]) # ...holding out 30% of the data for validation

if enable_telegram_bot:
    bot = telegram.Bot(token='591311395:AAEfSH464BdXSDezWGMZwdiLxLg2_aLlGDE')
    dir(bot)
    bot.send_message(chat_id=chat_id, text="{} - Training iniziato...".format(current_datetime()), timeout=100)

#configuring the custom callback for do the validation
validation_callback = ValidationCallback(X_validation, Y_validation, 5, chat_id=chat_id)

facesequence = FaceSequence(X_train, Y_train, batch_size, TRAINING_DATASET_FOLDER_NAME, epochs_with_same_data=epochs_with_same_data,
                            folders_at_the_same_time = folders_at_the_same_time,
                            to_avoid=validation_folders_list, enable_telegram_bot=enable_telegram_bot, chat_id=chat_id)

model.fit_generator(facesequence, epochs=num_epochs,
                    callbacks=[keras.callbacks.LambdaCallback(on_epoch_begin=lambda batch, logs: facesequence.on_epoch_begin()), validation_callback])
if enable_telegram_bot:
    bot = telegram.Bot(token='591311395:AAEfSH464BdXSDezWGMZwdiLxLg2_aLlGDE')
    bot.send_message(chat_id=chat_id, text="{} - Training completato!".format(current_datetime()), timeout=100)


# ONLY WHEN U WANT USE THE TEST SET!!!
# WARNING ONLY WHEN WE WANT THE TEST ERROR CAN BE DONE ONLY ONE TIME!
# loss = model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!


# ====== save model ======
# the three following instructions must be decommented when we want to save the model at the end of the training
if enable_telegram_bot:
    bot = telegram.Bot(token='591311395:AAEfSH464BdXSDezWGMZwdiLxLg2_aLlGDE')
    bot.send_message(chat_id=chat_id, text="{} - Sto salvando il modello".format(current_datetime()), timeout=100)

model.save('trained_model/{}.h5'.format(current_datetime()))

if enable_telegram_bot:
    bot.send_message(chat_id=chat_id, text="{} - Modello salvato!".format(current_datetime()), timeout=100)
