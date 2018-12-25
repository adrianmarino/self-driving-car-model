from keras import Sequential
from keras.layers import Lambda, Conv2D, Flatten, Dense, BatchNormalization, Input, Convolution2D, Dropout
from keras.layers.advanced_activations import ELU
from keras.models import Model
from keras.optimizers import Adam

from lib.model.metrics import rmse
from lib.model.model_wrapper import ModelWrapper


class ModelFactory:
    @staticmethod
    def create_nvidia_model(
            activation='relu',
            loss='mean_squared_error',
            metrics=[rmse],
            optimizer=Adam(lr=0.00001)
    ):
        # Inputs...
        image_input = Input(shape=(66, 200, 3), name='utils')

        # Normalise the utils - center the mean at 0
        image_input_net = Lambda(lambda img: (img / 255.0) - 0.5)(image_input)

        net = Conv2D(
            filters=24,
            kernel_size=(5, 5),
            strides=(2, 2),
            activation=activation,
            name='conv_1'
        )(image_input_net)
        net = BatchNormalization()(net)

        net = Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation=activation, name='conv_2')(net)
        net = BatchNormalization()(net)

        net = Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation=activation, name='conv_3')(net)
        net = BatchNormalization()(net)

        net = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation=activation, name='conv_4')(net)
        net = BatchNormalization()(net)

        net = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation=activation, name='conv_5')(net)
        net = BatchNormalization()(net)

        net = Flatten()(net)

        # Fully connected layers
        net = Dense(1164, activation=activation, name='dense_1')(net)
        net = BatchNormalization()(net)

        net = Dense(100, activation=activation, name='dense_3')(net)
        net = BatchNormalization()(net)

        net = Dense(50, activation=activation, name='dense_4')(net)
        net = BatchNormalization()(net)

        net = Dense(10, activation=activation, name='dense_5')(net)
        net = BatchNormalization()(net)

        # Outputs...
        steer_output = Dense(units=1, name='steer')(net)

        model = Model(inputs=[image_input], outputs=[steer_output])

        model.compile(
            loss=loss,
            loss_weights=[1],
            optimizer=optimizer,
            metrics=metrics
        )

        return ModelWrapper(model=model, model_name='NVidia Model 1')

    @staticmethod
    def create_commaai_model(
            loss='mean_squared_error',
            metrics=[rmse],
            optimizer=Adam(lr=1e-4)
    ):
        ch, row, col = 3, 66, 200  # camera format

        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1.,
                         #  input_shape=(ch, row, col),
                         #  output_shape=(ch, row, col)))
                         input_shape=(row, col, ch),
                         output_shape=(row, col, ch)))
        model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
        model.add(ELU())
        model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
        model.add(ELU())
        model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
        model.add(Flatten())
        model.add(Dropout(.2))
        model.add(ELU())
        model.add(Dense(512))
        model.add(Dropout(.5))
        model.add(ELU())
        model.add(Dense(1))

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return ModelWrapper(model=model, model_name='Comma AI Model')
