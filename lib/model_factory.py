from keras.layers import Lambda, Conv2D, Flatten, Dense, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from lib.model import Model


class ModelFactory:
    @staticmethod
    def create_nvidia_model(
            input_shape=(66, 200, 3),
            activation='relu',
            loss='mean_squared_error',
            optimizer=Adam(lr=0.001)
    ):
        model = Sequential()

        # Normalise the image - center the mean at 0
        model.add(Lambda(lambda imgs: (imgs / 255.0) - 0.5, input_shape=input_shape))

        model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation=activation))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation=activation))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation=activation))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation=activation))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation=activation))
        model.add(BatchNormalization())

        model.add(Flatten())

        # Fully connected layers
        model.add(Dense(1164, activation=activation))
        model.add(BatchNormalization())

        model.add(Dense(200, activation=activation))
        model.add(BatchNormalization())

        model.add(Dense(50, activation=activation))
        model.add(BatchNormalization())

        model.add(Dense(10, activation=activation))
        model.add(BatchNormalization())

        model.add(Dense(units=1))

        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        return Model(
            model=model,
            model_name='NVidia',
            description_image_path='./images/nvidia-model.png'
        )

