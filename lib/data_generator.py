import numpy as np
from keras.utils import Sequence

from lib.dataset.dataset import Dataset
from lib.augmentation.sample_augmenter import NullSampleAugmenter
import random


class SteeringWheelAngleDataGenerator(Sequence):
    def __init__(
            self,
            dataset,
            input_shapes,
            output_shape,
            batch_size,
            sample_augmenter,
            shuffle_per_epoch=False
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

        end_labels = []
        for label in labels.transpose():
            end_labels.append(label)

        return input_features, end_labels

    def __len__(self): return int(np.floor(len(self.dataset) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle_per_epoch:
            self.dataset = self.dataset.shuffle()

    def any_batch(self):
        features, labels = self[self.random_index()]
        return Dataset(features, ['image'], np.transpose(labels), self.dataset.label_columns)

    def random_index(self): return random.randint(0, len(self) - 1)
