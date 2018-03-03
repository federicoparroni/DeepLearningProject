
from LoadData import GetData
from LoadData import ResultPrediction

import numpy as np

from keras.utils import np_utils
from keras.models import load_model
import keras

import matplotlib.pyplot as plt
#===================================

bp = 'trained_model/'

NUM_CLASSES = 2
TEST_DATASET_FOLDER_NAME = '2_dataset test'

(X_test, y_test) = GetData(TEST_DATASET_FOLDER_NAME)

Y_test = np_utils.to_categorical(y_test, NUM_CLASSES)

model = load_model(bp + 'my_model.h5')

predicted_label = model.predict(X_test)

for i in range(1, 5):
    #print(np.squeeze(X_test[i], axis=0).shape)
    #plt.figure(1)
    result = (predicted_label[i])
    real_value = (Y_test[i])
    #plt.subplot(math.ceil(len(X_test/8)), 1, 1)
    plt.imshow(np.squeeze(X_test[i], axis=2), 'gray')
    plt.title(ResultPrediction(result, real_value))
    plt.show()