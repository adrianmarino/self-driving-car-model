import numpy as np
from keras.utils import Sequence

from lib.dataset import Dataset
from lib.sample_augmenter import NullSampleAugmenter
import random


class SteeringWheelAngleDataGenerator(Sequence):
    def __init__(
            self,
            dataset,
            input_shape,
            output_shape,
            batch_size,
            image_preprocessor,
            sample_augmenter=NullSampleAugmenter(),
            shuffle_per_epoch=False
    ):
        self.dataset = dataset
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.sample_augmenter = sample_augmenter
        self.image_preprocessor = image_preprocessor
        self.shuffle_per_epoch = shuffle_per_epoch
        self.on_epoch_end()

    def __getitem__(self, index):
        images = np.empty((self.batch_size, *self.input_shape))
        steers = np.empty((self.batch_size, *self.output_shape))
        samples_batch = self.dataset.subset(index, size=self.batch_size)

        index = 0
        for sample in samples_batch:
            image, steers[index] = self.sample_augmenter.augment(sample)
            images[index] = self.image_preprocessor.process(image)
            index += 1

        return images, steers

    def __len__(self): return int(np.floor(len(self.dataset) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle_per_epoch:
            self.dataset = self.dataset.shuffle()

    def any_batch(self):
        images, steers = self[self.random_index()]
        return Dataset(images, steers)

    def random_index(self): return random.randint(0, len(self) - 1)
