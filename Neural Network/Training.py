import numpy as np
import LoadData
from keras.utils import np_utils

from Utils import telegram_send_msg


def train(model, training_dataset_folder_name, epochs, batch_size, epochs_with_same_data,
          training_folders_count, validation_folders_count, validate_every, enable_telegram_bot=False):

    # load validation data that will be used to validate during the training
    validation_x, validation_y, validation_folders = load_data(folders=training_dataset_folder_name, folders_to_load=validation_folders_count)

    # train the model for the number of epochs specified
    for current_epoch in range(1, epochs+1):
        if (current_epoch-1) % epochs_with_same_data == 0:
            x, y, _ = load_data(training_dataset_folder_name, training_folders_count, to_avoid=validation_folders)

        model.fit(x, y, batch_size=batch_size, epochs=1, verbose=1, callbacks=None, shuffle=True)

        # perform validation
        if current_epoch % validate_every == 0:
            evaluation = model.evaluate(validation_x, validation_y,
                                        input_fn=None,
                                        feed_fn=None,
                                        batch_size=None,
                                        steps=None,
                                        metrics=None,
                                        name=None,
                                        checkpoint_path=None,
                                        hooks=None)
            if enable_telegram_bot:
                telegram_send_msg("Ho completato l'epoca {}\n{}\nValidation loss: {}, validation_accuracy: {}"
                                  .format(current_epoch, "-"*15, evaluation[0], evaluation[1]))

        if enable_telegram_bot:
            telegram_send_msg("Ho completato l'epoca {}".format(current_epoch))


def load_data(folders, folders_to_load=15, to_avoid=[]):
    x, y, loaded_folders_list = LoadData.GetData(folders, limit_value=folders_to_load, to_avoid=to_avoid)
    y = np_utils.to_categorical(y, 2)
    x = x.astype('float32')
    x /= np.max(x)
    return x, y, loaded_folders_list
