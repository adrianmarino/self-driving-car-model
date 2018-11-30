import numpy as np
from tensorflow.python.keras.utils import Sequence

from lib.image_preprocessor import ImagePreprocessor
from lib.sample_augmenter import NullSampleAugmenter


class SteeringWheelAngleDataGenerator(Sequence):
    def __init__(
            self,
            dataset,
            input_shape,
            output_shape,
            batch_size,
            sample_augmenter=NullSampleAugmenter(),
            image_preprocessor=ImagePreprocessor()
    ):
        self.dataset = dataset
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.sample_augmenter = sample_augmenter
        self.image_preprocessor = image_preprocessor
        self.on_epoch_end()

    def __getitem__(self, index):
        images = np.empty(self.batch_size, *self.input_shape)
        steers = np.empty(self.batch_size, *self.output_shape)

        for index, sample in iter(self.dataset.subset(index, size=self.batch_size)):
            image, steers[index] = self.sample_augmenter.augment(sample)
            images[index] = self.image_preprocessor.process(image)

        return images, steers

    def on_epoch_end(self): self.dataset = self.dataset.shuffle()

    def __len__(self): return int(np.floor(len(self.dataset) / self.batch_size))


