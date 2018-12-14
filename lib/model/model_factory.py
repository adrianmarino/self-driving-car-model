from keras.layers import Lambda, Conv2D, Flatten, Dense, BatchNormalization, Input, Reshape
from keras.models import Model
from keras.optimizers import Adam

from lib.model.model_wrapper import ModelWrapper


class ModelFactory:
    @staticmethod
    def create_nvidia_model(
            input_shape=(66, 200, 3),
            activation='relu',
            loss='mean_squared_error',
            metrics=[],
            optimizer=Adam(lr=0.0001)
    ):
        inputs = Input(shape=input_shape)

        # Normalise the image - center the mean at 0
        net = Lambda(lambda imgs: (imgs / 255.0) - 0.5)(inputs)

        net = Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation=activation, name='conv_1')(net)
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

        net = Dense(200, activation=activation, name='dense_2')(net)
        net = BatchNormalization()(net)

        net = Dense(50, activation=activation, name='dense_3')(net)
        net = BatchNormalization()(net)

        net = Dense(10, activation=activation, name='dense_4')(net)
        net = BatchNormalization()(net)

        steering_angle_output = Dense(units=1, name='steering_angle')(net)

        throttle_output = Dense(units=1, name='throttle')(net)

        model = Model(inputs=inputs, outputs=[steering_angle_output, throttle_output])

        model.compile(
            loss=loss,
            loss_weights=[0.7, 0.3],
            optimizer=optimizer,
            metrics=metrics
        )

        return ModelWrapper(model=model, model_name='NVidia')
