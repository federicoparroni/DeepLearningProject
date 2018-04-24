import keras.callbacks
import telegram

from Utils import current_datetime


class ValidationCallback(keras.callbacks.Callback):

    def __init__(self, x_val, y_val, validate_every=5, enable_telegram_bot=True, chat_id="undefined"):
        self.counter = 1
        self.val_history_index = 0
        self.validation_history = []
        self.x = x_val
        self.y = y_val
        self.validate_every = validate_every
        self.enable_telegram_bot = enable_telegram_bot
        self.chat_id = chat_id

    def on_epoch_end(self, epoch, logs=None):

        if self.counter % self.validate_every == 0:

            print('\n\n\n ===== Validation =====')
            evaluation = self.model.evaluate(self.x, self.y)

            if self.enable_telegram_bot:
                bot = telegram.Bot(token='591311395:AAEfSH464BdXSDezWGMZwdiLxLg2_aLlGDE')
                message = "{}\nValidation loss: {}\nValidation accuracy: {}".format(current_datetime(),
                                                                                     evaluation[0], evaluation[1])
                bot.send_message(chat_id=self.chat_id, text=message, timeout=100)

            self.validation_history.append(evaluation)
            print('\nval_loss -> {} \nval_accuracy -> {}'.format(self.validation_history[self.val_history_index][0],
                                                              self.validation_history[self.val_history_index][1]))
            self.val_history_index += 1

        self.counter += 1
