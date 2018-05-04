from Utils import current_datetime
import telegram
import keras
import numpy as np
import LoadData
import threading
from keras.utils import np_utils
from Utils import telegram_send_msg

_instances = {}


class SingletonTrain(object):

    def __new__(cls, *args, **kw):
        if not cls in _instances:
            instance = super().__new__(cls)
            _instances[cls] = instance

        return _instances[cls]

    def Train_Sequence(self, model, train_sequence, num_epochs=200, chat_id="undefined", training_callbacks=[], save_model=True):
        if chat_id != "undefined":
            telegram_send_msg("Training iniziato...")

        model.fit_generator(train_sequence, epochs=num_epochs, callbacks=[keras.callbacks.LambdaCallback(
                            on_epoch_begin=lambda batch, logs: train_sequence.on_epoch_begin())]+training_callbacks)

        if chat_id != "undefined":
            telegram_send_msg("Training completato!")

        # ONLY WHEN U WANT USE THE TEST SET!!!
        # WARNING ONLY WHEN WE WANT THE TEST ERROR CAN BE DONE ONLY ONE TIME!
        # loss = model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!

        # ====== save model ======
        # the three following instructions must be decommented when we want to save the model at the end of the training
        if chat_id != "undefined" and save_model:
            telegram_send_msg("Sto salvando il modello")

        if save_model:
            model.save('trained_model/{}.h5'.format(current_datetime()))

        if chat_id != "undefined" and save_model:
            telegram_send_msg("{} - Modello salvato!")

    y_next_epoch = []
    x_next_epoch = []

    def Train(self, model, training_dataset_folder_name, epochs, batch_size, epochs_with_same_data,
              training_folders_count, validation_x, validation_y, to_avoid, validate_every, class_weight,
              enable_telegram_bot=False, save_model=True):
        t = None
        validation_history = []

        # load validation data that will be used to validate during the training

        x, y, _ = self.load_data(folders=training_dataset_folder_name, folders_to_load=5,
                                 to_avoid=to_avoid)

        # train the model for the number of epochs specified
        for current_epoch in range(epochs):
            # print("\nactually running epoch " + str(current_epoch))
            if current_epoch % epochs_with_same_data == 0:
                # print("\ncurrent_epoch % epochs_with_same_data = 0, i will fetch data for the next epoch")
                t = threading.Thread(target=self.load_data, args=(training_dataset_folder_name, training_folders_count,
                                                                  to_avoid))
                t.setDaemon(True)
                t.start()

            print("\nEpoch {}/{}".format(current_epoch+1, epochs))
            model.fit(x, y, batch_size=batch_size, epochs=1, verbose=1, class_weight=class_weight, callbacks=None,
                      shuffle=True)

            # perform validation
            if (current_epoch+1) % validate_every == 0:
                evaluation = model.evaluate(validation_x, validation_y)
                print(evaluation)
                validation_history.append(evaluation)
                if enable_telegram_bot:
                    telegram_send_msg("Ho completato l'epoca {}\n{}\nValidation loss: {}, validation_accuracy: {}"
                                      .format(current_epoch+1, "-"*15, evaluation[0], evaluation[1]))

            if enable_telegram_bot:
                telegram_send_msg("Ho completato l'epoca {}".format(current_epoch+1))

            if (current_epoch + 1) % epochs_with_same_data == 0:
                t.join()
                x = self.x_next_epoch
                y = self.y_next_epoch

        if save_model:
            model.save('trained_model/{}.h5'.format(current_datetime()))

        return validation_history


    def load_data(self, folders, folders_to_load=15, to_avoid=[]):
        # print("\nstarted the fetch of data for the next epoch in parallel")
        self.x_next_epoch, self.y_next_epoch, loaded_folders_list = LoadData.GetData(folders,limit_value=folders_to_load,
                                                                                     to_avoid=to_avoid)
        self.y_next_epoch = np_utils.to_categorical(self.y_next_epoch, 2)
        self.x_next_epoch = self.x_next_epoch.astype('float32')
        self.x_next_epoch /= np.max(self.x_next_epoch)
        # print("\nended the fetch of data for the next epoch in parallel")
        return self.x_next_epoch, self.y_next_epoch, loaded_folders_list
