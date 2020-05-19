# Carregar o modelo
from keras import backend as K,  regularizers
from keras.models import load_model as keras_load_model, Sequential
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD, Adam


def global_average_pooling(x):
    # Mean of a tensor, alongside the specified axis.
    # Reference: https://www.tensorflow.org/api_docs/python/tf/keras/backend/mean
    return K.mean(x, axis=(2, 3))


def global_average_pooling_shape(input_shape):
    return input_shape[0:2]


def convolutions():
    dropout = 0.15
    bn_momentum = 0.4
    l2 = 0.0001
    # Model architecture definition
    model = Sequential()

    model.add(Convolution2D(32, (7, 7), activation='relu', input_shape=(
        224, 224, 3), kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization(momentum=bn_momentum))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))

    model.add(Convolution2D(32, (7, 7), activation='relu',
                            kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization(momentum=bn_momentum))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Convolution2D(64, (5, 5), activation='relu',
                            kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization(momentum=bn_momentum))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))

    model.add(Convolution2D(64, (5, 5), activation='relu',
                            kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization(momentum=bn_momentum))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Convolution2D(128, (3, 3), activation='relu',
                            kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization(momentum=bn_momentum))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(dropout))

    model.add(Convolution2D(128, (3, 3), activation='relu',
                            kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization(momentum=bn_momentum))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    # this converts our 3D feature maps to 1D feature vectors
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))

    # Output Layer
    model.add(Dense(2, activation='softmax'))  # antes softmax
    return model


def compile_model(model, optimizer_='adam'):
    l_r = 0.001
    if optimizer_ == "sgd":
        model.compile(loss='categorical_crossentropy',
                      # antes sgd, adam, rmsprop
                      optimizer=SGD(lr=l_r, decay=1e-6),
                      metrics=['accuracy'])
    if optimizer_ == "adam":
        model.compile(loss='categorical_crossentropy',
                      # antes sgd, adam, rmsprop
                      optimizer=Adam(lr=l_r, decay=1e-6),
                      metrics=['accuracy'])
    return model


def load_model(model_path, optimizer_='adam'):
    '''
    Load model
    model_path: Path to the model file
    '''
    model = keras_load_model(model_path)
    model.load_weights(model_path)
    model = compile_model(model, optimizer_)

    return model


def get_model(optimizer_='adam'):
    model = convolutions()
    model = compile_model(model, optimizer_)
    return model


def get_model_layers(model):
    '''
    Get a list with model's layers names

    '''
    return list(dict([(layer.name, layer) for layer in model.layers]).keys())



def get_model_viewable_layers(model):
    '''
    Get a list with model's viewable layers names

    '''
    return list(dict([(layer.name, layer) for layer in model.layers if len(layer.output_shape) == 4]).keys())


def get_model_nb_classes(model):
    '''
    Get number of classes from a model

    '''
    return model.layers[-1].output_shape[1]
