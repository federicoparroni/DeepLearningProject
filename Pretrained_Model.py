from LoadData import GetData
from LoadData import ResultPrediction

from PlotImage import plot

from keras.utils import np_utils
from keras.models import load_model


#===================================

bp = 'trained_model/'

NUM_CLASSES = 2
TEST_DATASET_FOLDER_NAME = '2_dataset test'
MAX_IMAGES_TO_PLOT = 49

model = load_model(bp + 'my_model.h5')

(X_test, y_test) = GetData(TEST_DATASET_FOLDER_NAME)
Y_test = np_utils.to_categorical(y_test, NUM_CLASSES)

predicted_label = model.predict(X_test)

titles = []
for i in range(len(X_test)):
    result = (predicted_label[i])
    real_value = (Y_test[i])
    # plt.imshow(np.squeeze(X_test[i], axis=2), 'gray')
    titles.append(ResultPrediction(result, real_value))

plot(X_test, titles, MAX_IMAGES_TO_PLOT)
