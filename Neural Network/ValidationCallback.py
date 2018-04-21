import keras.callbacks


class ValidationCallback(keras.callbacks.Callback):

    def __init__(self, x_val, y_val, validate_every):
        self.counter = 1
        self.validation_history = []
        self.x = x_val
        self.y = y_val
        self.validate_every = validate_every

    def on_epoch_end(self, epoch, logs=None):

        if self.counter % self.validate_every == 0:
            a = self.model.evaluate(self.x, self.y)
            self.validation_history.append(a)

        self.counter += 1
