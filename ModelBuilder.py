from keras.models import Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, Activation


class ModelBuilder:

    """
    input for the method is a vector []
    in each position there is another vector with one of the following structure:
        'convolution', conv_depth, kernel_size, activation
        'pooling', pool_size
        'dropout', dropout_probability
        'flatten'
        'dense', hidden_size, activation
    """

    def __init__(self, model_structure, input_shape):
        model = Sequential()
        for i in range(0, len(model_structure)-1):
            ms = model_structure[i]
            if ms[0] == 'conv':
                if i == 0:
                    if len(ms) == 4:
                        model.add(Convolution2D(ms[1], (ms[2], ms[2]), input_shape=input_shape))
                        model.add(Activation(ms[3]))
                    else:
                        model.add(Convolution2D(ms[1], (ms[2], ms[2]), input_shape=input_shape))
                        model.add(Activation('relu'))
                else:
                    if len(ms) == 4:
                        model.add(Convolution2D(ms[1], (ms[2], ms[2])))
                        model.add(Activation(ms[3]))
                    else:
                        model.add(Convolution2D(ms[1], (ms[2], ms[2])))
                        model.add(Activation('relu'))
            elif ms[0] == 'pool':
                model.add(MaxPooling2D(pool_size=(ms[1], ms[1])))
            elif ms[0] == 'dropout':
                model.add(Dropout(ms[1]))
            elif ms[0] == 'flatten':
                model.add(Flatten())
            elif ms[0] == 'dense':
                if len(ms) == 3:
                    model.add(Dense(ms[1]))
                    model.add(Activation(ms[2]))
                else:
                    model.add(Dense(ms[1]))
                    model.add(Activation('relu'))
        self.model = model

a = [['conv', 16, 3], ['pool', 3], ['flatten'], ['dense', 128], ['dense', 128]]
size = (80, 80, 2)
modelObject = ModelBuilder(a, size)
print(modelObject.model.summary())
