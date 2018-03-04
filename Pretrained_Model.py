from LoadData import GetData
from LoadData import ResultPrediction

from PlotImage import plot

from keras.utils import np_utils
from keras.models import load_model


#===================================

bp = 'trained_model/'

NUM_CLASSES = 2
TEST_DATASET_FOLDER_NAME = '3_preprocessed_2_dataset test'
MAX_IMAGES_TO_PLOT = 36
NUM_PRINTED_PAGES = 3
MODEL_TO_LOAD = '2018-03-04 17:54:14.h5'

model = load_model(bp + MODEL_TO_LOAD)

(X_test, y_test) = GetData(TEST_DATASET_FOLDER_NAME)
Y_test = np_utils.to_categorical(y_test, NUM_CLASSES)

predicted_label = model.predict(X_test)

# creates titles to plot the predicted classes
titles = []
for i in range(len(X_test)):
    result = (predicted_label[i])
    real_value = (Y_test[i])
    # plt.imshow(np.squeeze(X_test[i], axis=2), 'gray')
    titles.append(ResultPrediction(result, real_value))

# show the images in plots
for k in range(NUM_PRINTED_PAGES):
    _from = k*MAX_IMAGES_TO_PLOT
    _to = (k+1)*MAX_IMAGES_TO_PLOT
    plot(X_test[_from:_to], titles[_from:_to], MAX_IMAGES_TO_PLOT)
