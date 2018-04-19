import keras
import numpy as np
import LoadData
from keras.utils import np_utils

class FaceSequence(keras.utils.Sequence):


    def __init__(self, x_set, y_set, batch_size, training_dataset_folder_name, epochs_with_same_data = 5, folders_at_the_same_time = 20, to_avoid = []):
        self.x, self.y = x_set, y_set
        self.epoch = 0
        self.batch_size = batch_size
        self.epochs_with_same_data = epochs_with_same_data
        self.training_dataset_folder_name = training_dataset_folder_name
        self.folders_at_the_same_time = folders_at_the_same_time
        self.to_avoid = to_avoid

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        self.epoch += 1
        if self.epoch % self.epochs_with_same_data == 0:
            self.x, self.y, _ = LoadData.GetData(self.training_dataset_folder_name, limit_value=self.folders_at_the_same_time, to_avoid = self.to_avoid)
            self.y = np_utils.to_categorical(self.y, 2)
            self.x = self.x.astype('float32')
            self.x /= np.max(self.x)
        else:
            s = np.arange(self.x.shape[0])
            np.random.shuffle(s)
            self.x = self.x[s]
            self.y = self.y[s]