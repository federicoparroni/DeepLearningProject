from LoadData import GetData
from LoadData import ResultPrediction

from PlotImage import plot

from keras.utils import np_utils
from keras.models import load_model
import numpy as np


#===================================

bp = 'trained_model/'

NUM_CLASSES = 2
TEST_DATASET_FOLDER_NAME = '3_preprocessed_2_dataset test'
MAX_IMAGES_TO_PLOT = 36
NUM_PRINTED_PAGES = 3
MODEL_TO_LOAD = 'ilToro.h5'

model = load_model(bp + MODEL_TO_LOAD)

(X_test, y_test) = GetData(TEST_DATASET_FOLDER_NAME)
Y_test = np_utils.to_categorical(y_test, NUM_CLASSES)

sum = 0
for i in range(len(y_test)):
    sum += y_test[i]

print(sum)
print(len(y_test))

predicted_label = model.predict(X_test)

# creates titles to plot the predicted classes
titles = []
for i in range(len(X_test)):
    result = predicted_label[i]
    real_value = Y_test[i]
    # plt.imshow(np.squeeze(X_test[i], axis=2), 'gray')
    titles.append(ResultPrediction(result, real_value))


X_test_concatenated = []
for i in range(NUM_PRINTED_PAGES*MAX_IMAGES_TO_PLOT):
    X_test_concatenated.append(np.concatenate((np.array(X_test[i, :, :, 0]), np.array(X_test[i, :, :, 1]))))
    #print(np.array(X_test_concatenated).shape)

# show the images in plots
for k in range(NUM_PRINTED_PAGES):
    _from = k*MAX_IMAGES_TO_PLOT
    _to = (k+1)*MAX_IMAGES_TO_PLOT
    plot(X_test_concatenated[_from:_to], titles[_from:_to], MAX_IMAGES_TO_PLOT)
