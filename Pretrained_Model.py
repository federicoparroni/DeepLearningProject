from LoadData import GetData
from LoadData import ResultPrediction

from PlotImage import plot

from keras.utils import np_utils
from keras.models import load_model
import keras

#===================================

bp = 'trained_model/'

NUM_CLASSES = 2
TEST_DATASET_FOLDER_NAME = '2_dataset test'

model = load_model(bp + 'my_model.h5')

(X_test, y_test) = GetData(TEST_DATASET_FOLDER_NAME)
Y_test = np_utils.to_categorical(y_test, NUM_CLASSES)

predicted_label = model.predict(X_test)

titles = []
for i in range(1, 5):
    result = (predicted_label[i])
    real_value = (Y_test[i])
    # plt.imshow(np.squeeze(X_test[i], axis=2), 'gray')
    titles.append(ResultPrediction(result, real_value))

plot(X_test, titles)
