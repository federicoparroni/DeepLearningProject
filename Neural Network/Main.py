import numpy as np
import tensorflow as tf
import keras.callbacks
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

from Utils import current_datetime

import telegram

from LoadData import GetData
from FaceSequence import FaceSequence
from ValidationCallback import  ValidationCallback


# ====================CONFIGURING GPU ========================================
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# ============================================================================
enable_telegram_bot = True
# chat_id = 125016709               # this is my private chat id
# chat_id = "@gdptensorboard"       # this is the name of the public channel
chat_id = -1001223624517            # this is for the private channel

# defining the folders path train and test
TRAINING_DATASET_FOLDER_NAME = '3_preprocessed_1_dataset train'
TEST_DATASET_FOLDER_NAME = '3_preprocessed_2_dataset test'

epochs_with_same_data = 3

folders_at_the_same_time = 15
validation_folders = 12

batch_size = 128            # in each iteration, we consider 128 training examples at once
num_epochs = 180            # we iterate 200 times over the entire training set
kernel_size = 3             # we will use 3x3 kernels throughout
pool_size = 2               # we will use 2x2 pooling throughout
conv_depth_1 = 16           # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 16           # ...switching to 64 after the first pooling layer
drop_prob_conv = 0.1        # dropout after pooling with probability 0.25
drop_prob_hidden = 0.3      # dropout in the FC layer with probability 0.5
hidden_size = 128           # the FC layer will have 512 neurons

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


inp = Input(shape=(height, width, depth))

# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
drop_1 = Dropout(drop_prob_conv)(pool_1)
conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_2 = Dropout(drop_prob_conv)(pool_2)
conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_2)
pool_3 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_3)
drop_3 = Dropout(drop_prob_conv)(pool_3)
conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_3)


# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
flat = Flatten()(conv_4)

hidden = Dense(hidden_size, activation='relu')(flat)
drop_4 = Dropout(drop_prob_hidden)(hidden)
hidden2 = Dense(hidden_size, activation='relu')(drop_4)
drop_5 = Dropout(drop_prob_hidden)(hidden2)
hidden3 = Dense(hidden_size, activation='relu')(hidden2)

out = Dense(num_classes, activation='softmax')(hidden3)

model = Model(inputs=inp, outputs=out)      # To define a model, just specify its input and output layers

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
validation_callback = ValidationCallback(X_validation, Y_validation, 5)

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
