from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, Convolution3D, MaxPooling3D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import keras
from LoadData import GetData
import numpy as np
import time
import datetime
import tensorflow as tf

#====================CONFIGURING GPU ========================================

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#============================================================================

#defining the folders path train and test
TRAINING_DATASET_FOLDER_NAME = '3_preprocessed_1_dataset train'
TEST_DATASET_FOLDER_NAME = '3_preprocessed_2_dataset test'

batch_size = 128 # in each iteration, we consider 32 training examples at once
num_epochs = 1 # we iterate 200 times over the entire training set
kernel_size = 2 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth_1 = 2 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 2 # ...switching to 64 after the first pooling layer
drop_prob_conv = 0.1 # dropout after pooling with probability 0.25
drop_prob_hidden = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 2 # the FC layer will have 512 neurons

## we were loading training and test data like this before ##
(X_train, y_train), (X_test, y_test) = (GetData(TRAINING_DATASET_FOLDER_NAME), GetData(TEST_DATASET_FOLDER_NAME)) # fetch data


num_train, height, width, depth = X_train.shape
# num_test = X_test.shape[0] #num test images
num_classes = 2 # there are 2 image classes

X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
X_train /= np.max(X_train) # Normalise data to [0, 1] range
# X_test /= np.max(X_test) # Normalise data to [0, 1] range

Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
# Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels


inp = Input(shape=(height,width, depth))

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

#===========================
class MetricsRetrieving(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.metrics = []

    def on_epoch_end(self, epoch, logs={}):
        self.metrics.append(logs.get('accuracy'))

metrics = MetricsRetrieving()


#==========================


model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

#INSTRUCTION TO ABLE THE EARLY STOPPING
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=1, verbose=1, mode='auto')

#====== configuring tensorboard ======
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# a = model.fit(X_train, Y_train,   # Train the model using the training set...
#           batch_size=batch_size, epochs=num_epochs,
#           verbose=1, validation_split=0.3, callbacks=[tbCallBack, earlyStopping])
#print(a.history)


#k-folds crossvalidation
def K_Fold_CrossValidation(X_Train, Y_Train, K):
    inputs_folds = np.array_split(X_Train, K)
    outputs_folds = np.array_split(Y_Train, K)
    folds_val_acc = []
    for i in range(len(inputs_folds)):

        validation = [inputs_folds[i], outputs_folds[i]]

        print(np.array(inputs_folds).shape)
        print(np.array(inputs_folds[:i]).shape)
        print(np.array(inputs_folds[i+1:]).shape)

        train_inp = np.concatenate((inputs_folds[:i], inputs_folds[i+1:]))
        train_out = np.concatenate((outputs_folds[:i], outputs_folds[i+1:]))

        fold_i = model.fit(train_inp, train_out,
                      batch_size=batch_size, epochs=num_epochs,
                      verbose=1, validation_data=validation, shuffle='true', callbacks=[tbCallBack, earlyStopping])
        folds_val_acc.append(fold_i.history.get('val_acc'))
        k_fold_acc = np.sum(folds_val_acc) / K
        print(k_fold_acc)


K_Fold_CrossValidation(X_train, Y_train, 5)




# ONLY WHEN U WANT USE THE TEST SET!!!
# WARNING ONLY WHEN WE WANT THE TEST ERROR CAN BE DONE ONLY ONE TIME!
#loss = model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!


#====== save model ======
#the three following instructions must be decommented when we want to save the model at the end of the training

#ts = time.time()
#st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

#model.save('trained_model/' + st + '.h5')