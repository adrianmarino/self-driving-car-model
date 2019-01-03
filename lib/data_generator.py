import numpy as np
from keras.utils import Sequence

from lib.dataset.dataset import Dataset
from lib.augmentation.sample_augmenter import NullSampleAugmenter
import random


class DataGenerator(Sequence):

    @staticmethod
    def create_from(train_set, train_augmenter, cfg, shuffle_per_epoch=True):
        image_input_shape = [
            (
                cfg['network.image_input_shape.height'],
                cfg['network.image_input_shape.width'],
                cfg['network.image_input_shape.channels']
            )
        ]
        output_shape = (1,)

        return DataGenerator(
            train_set,
            image_input_shape,
            output_shape,
            cfg['train.batch_size'],
            train_augmenter,
            shuffle_per_epoch
        )

    def __init__(
            self,
            dataset,
            input_shapes,
            output_shape,
            batch_size,
            sample_augmenter,
            shuffle_per_epoch=True
    ):
        self.dataset = dataset
        self.input_shapes = input_shapes
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.sample_augmenter = sample_augmenter
        self.shuffle_per_epoch = shuffle_per_epoch
        self.on_epoch_end()

    def __getitem__(self, index):
        input_features = [np.empty((self.batch_size, *input_shape)) for input_shape in self.input_shapes]
        labels = np.empty((self.batch_size, *self.output_shape))
        samples_batch = self.dataset.subset(index, size=self.batch_size)

        index = 0
        for sample in samples_batch:
            augment_sample = self.sample_augmenter.augment(sample)

            for augmented_features_index, augmented_features in enumerate(augment_sample.features):
                input_features[augmented_features_index][index] = augmented_features

            labels[index] = augment_sample.labels

            index += 1

        labels = [label for label in labels.transpose()]

        return input_features, labels

    def __len__(self): return int(np.floor(len(self.dataset) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle_per_epoch:
            self.dataset = self.dataset.shuffle()

    def any_batch(self):
        features, labels = self[self.random_index()]
        return Dataset(features, ['utils'], np.transpose(labels), self.dataset.label_columns)

    def random_index(self): return random.randint(0, len(self) - 1)
