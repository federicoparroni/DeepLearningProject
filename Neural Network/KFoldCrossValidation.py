import os
import numpy as np
from LoadData import GetData
from keras.utils import np_utils
import Train
from Utils import current_datetime
import keras


def CrossValidate(k, models, dataset_folder_name, batch_size, num_epochs=200, chat_id="undefined",
                  folders_at_the_same_time=20, max_num_of_validation_folders=12, epochs_with_same_data=5,
                  validate_every=5):
    avg_val_accuracy_models = []
    total_num_folders = len(os.listdir(dataset_folder_name))
    folders_each_validation = total_num_folders // k if total_num_folders < max_num_of_validation_folders else max_num_of_validation_folders

    with open('crossvaliationresults.txt', 'w') as the_file:
        the_file.write(current_datetime())

    for i in range(len(models)):
        print("\n validating model: " + str(i))
        sum_model_validations_acc = 0
        to_avoid_validation = []
        for j in range(k):
            print("\n validation round " + str(j))
            (X_validation, Y_validation, validation_folders_list) = GetData(dataset_folder_name,
                                                                            limit_value=folders_each_validation)
            X_validation = X_validation.astype('float32')
            X_validation /= np.max(X_validation)
            Y_validation = np_utils.to_categorical(Y_validation, 2)
            to_avoid_validation = to_avoid_validation + validation_folders_list

            # print("\n next validation folders wont be fetched from: ")
            # print(*to_avoid_validation, sep=' ')

            # validation_callback = ValidationCallback(X_validation, Y_validation, chat_id=chat_id,
            #                                          enable_telegram_bot=False if chat_id == "undefined" else True,
            #                                          validate_every=validate_every)
            # facesequence = FaceSequence(batch_size, dataset_folder_name, num_epochs,
            #                             epochs_with_same_data=epochs_with_same_data,
            #                             folders_at_the_same_time=folders_at_the_same_time,
            #                             to_avoid=validation_folders_list,
            #                             enable_telegram_bot=False if chat_id == "undefined" else True,
            #                             chat_id=chat_id)
            #
            # # do the actual training
            # Train(models[i], facesequence, num_epochs, chat_id=chat_id,
            #       training_callbacks=training_callbacks + [-validation_callback], save_model=save_each_model)

            validation_history = Train.SingletonTrain().Train(models[i], training_dataset_folder_name=dataset_folder_name, epochs=num_epochs,
                                                              batch_size=batch_size, epochs_with_same_data=epochs_with_same_data,
                                                              training_folders_count=folders_at_the_same_time, validation_x= X_validation,
                                                              validation_y=Y_validation, to_avoid=validation_folders_list,
                                                              validate_every=validate_every, enable_telegram_bot=False if chat_id == "undefined" else True)

            sum_model_validations_acc += (validation_history[-1])[1]

        avg_val_accuracy_models = avg_val_accuracy_models + [sum_model_validations_acc / k]

        with open('crossvaliationresults.txt', 'a') as the_file:
            the_file.write(keras.utils.print_summary(models[i], line_length=None, positions=None, print_fn=None))
            the_file.write('\n validation results ' + str(sum_model_validations_acc / k))
    a = 1