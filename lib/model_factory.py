from keras.layers import Lambda, Conv2D, Dropout, Flatten, Dense
from keras.models import Sequential

from lib.model import Model


class ModelFactory:
    @staticmethod
    def create_nvidia_model(
            input_shape=(160, 320, 3),
            input_normalization=lambda x: x / 127.5 - 1.0,
            cnn_end_dropout_rate=0.5,
            activation='elu'
    ):
        model = Sequential()
        model.add(Lambda(input_normalization, input_shape=input_shape))

        model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation=activation))
        model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation=activation))
        model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation=activation))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation=activation))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation=activation))

        model.add(Dropout(cnn_end_dropout_rate))

        model.add(Flatten())

        model.add(Dense(units=100, activation=activation))
        model.add(Dense(units=50, activation=activation))
        model.add(Dense(units=10, activation=activation))

        # Default activation is applied (ie. "linear" activation: a(x) = x).
        model.add(Dense(units=1))

        return Model(
            model=model,
            model_name='NVidia',
            description_image_path='./images/nvidia-model.png'
        )
