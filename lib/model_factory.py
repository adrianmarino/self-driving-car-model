from keras.layers import Lambda, Conv2D, Flatten, Dense, BatchNormalization
from keras.models import Sequential

from lib.model import Model


class ModelFactory:
    @staticmethod
    def create_nvidia_model(activation='relu'):
        model = Sequential()

        # Cropping image
        model.add(Lambda(lambda imgs: imgs[:, 80:, :, :], input_shape=(160, 320, 3)))

        # Normalise the image - center the mean at 0
        model.add(Lambda(lambda imgs: (imgs / 255.0) - 0.5))

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
        model.add(Dense(1024, activation=activation))
        model.add(BatchNormalization())

        model.add(Dense(512, activation=activation))
        model.add(BatchNormalization())

        model.add(Dense(128, activation=activation))
        model.add(BatchNormalization())

        model.add(Dense(64, activation=activation))
        model.add(BatchNormalization())

        model.add(Dense(32, activation=activation))
        model.add(BatchNormalization())

        model.add(Dense(16, activation=activation))
        model.add(BatchNormalization())

        model.add(Dense(units=1))

        return Model(
            model=model,
            model_name='NVidia' #,
           # description_image_path='./images/nvidia-model.png'
        )

