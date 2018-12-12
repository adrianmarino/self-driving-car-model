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
        features = np.empty((self.batch_size, *self.input_shape))
        labels = np.empty((self.batch_size, *self.output_shape))
        samples_batch = self.dataset.subset(index, size=self.batch_size)

        index = 0
        for sample in samples_batch:
            image, labels[index][0], labels[index][1], labels[index][2] = self.sample_augmenter.augment(sample)
            features[index] = self.image_preprocessor.process(image)
            index += 1

        return features, labels

    def __len__(self): return int(np.floor(len(self.dataset) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle_per_epoch:
            self.dataset = self.dataset.shuffle()

    def any_batch(self):
        features, labels = self[self.random_index()]
        return Dataset(features, ['image'], labels, self.dataset.label_columns)

    def random_index(self): return random.randint(0, len(self) - 1)
