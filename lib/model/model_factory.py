from keras.layers import Lambda, Conv2D, Flatten, Dense, BatchNormalization, Input, concatenate
from keras.models import Model
from keras.optimizers import Adam

from lib.model.model_wrapper import ModelWrapper


class ModelFactory:
    @staticmethod
    def create_nvidia_model(
            activation='relu',
            loss='mean_squared_error',
            metrics=[],
            optimizer=Adam(lr=0.0001)
    ):
        # Inputs...
        image_input = Input(shape=(66, 200, 3), name='image')
        speed_input = Input(shape=(1,), name='speed')
        reverse_input = Input(shape=(1,), name='reverse')

        # Normalise the image - center the mean at 0
        image_input_net = Lambda(lambda img: (img / 255.0) - 0.5)(image_input)    
        speed_input_net = Lambda(lambda speed: (speed / 32.0) - 0.5)(speed_input)
        
        net = Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation=activation, name='conv_1')(image_input_net)
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

        net = concatenate([net, speed_input_net, reverse_input])

        # Fully connected layers
        net = Dense(1500, activation=activation, name='dense_1')(net)
        net = BatchNormalization()(net)

        net = Dense(600, activation=activation, name='dense_2')(net)
        net = BatchNormalization()(net)

        net = Dense(200, activation=activation, name='dense_3')(net)
        net = BatchNormalization()(net)

        net = Dense(100, activation=activation, name='dense_4')(net)
        net = BatchNormalization()(net)

        net = Dense(50, activation=activation, name='dense_5')(net)
        net = BatchNormalization()(net)

        # Outputs...
        steer_output = Dense(units=1, name='steer')(net)
        throttle_output = Dense(units=1, name='throttle')(net)

        model = Model(inputs=[image_input, speed_input, reverse_input], outputs=[steer_output, throttle_output])

        model.compile(
            loss=loss,
            loss_weights=[1.5, 1],
            optimizer=optimizer,
            metrics=metrics
        )

        return ModelWrapper(model=model, model_name='NVidia')
