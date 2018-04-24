from Utils import current_datetime
import telegram
import keras


def Train(model, train_sequence, num_epochs=200, chat_id="undefined", training_callbacks=[], save_model=True):
    if chat_id != "undefined":
        bot = telegram.Bot(token='591311395:AAEfSH464BdXSDezWGMZwdiLxLg2_aLlGDE')
        dir(bot)
        bot.send_message(chat_id=chat_id, text="{} - Training iniziato...".format(current_datetime()), timeout=100)

    model.fit_generator(train_sequence, epochs=num_epochs, callbacks=[keras.callbacks.LambdaCallback(
                        on_epoch_begin=lambda batch, logs: train_sequence.on_epoch_begin())]+training_callbacks)

    if chat_id != "undefined":
        bot = telegram.Bot(token='591311395:AAEfSH464BdXSDezWGMZwdiLxLg2_aLlGDE')
        bot.send_message(chat_id=chat_id, text="{} - Training completato!".format(current_datetime()), timeout=100)

    # ONLY WHEN U WANT USE THE TEST SET!!!
    # WARNING ONLY WHEN WE WANT THE TEST ERROR CAN BE DONE ONLY ONE TIME!
    # loss = model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!

    # ====== save model ======
    # the three following instructions must be decommented when we want to save the model at the end of the training
    if chat_id != "undefined" and save_model:
        bot = telegram.Bot(token='591311395:AAEfSH464BdXSDezWGMZwdiLxLg2_aLlGDE')
        bot.send_message(chat_id=chat_id, text="{} - Sto salvando il modello".format(current_datetime()), timeout=100)

    if save_model:
        model.save('trained_model/{}.h5'.format(current_datetime()))

    if chat_id != "undefined" and save_model:
        bot.send_message(chat_id=chat_id, text="{} - Modello salvato!".format(current_datetime()), timeout=100)
