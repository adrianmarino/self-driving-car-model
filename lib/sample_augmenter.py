import numpy as np

from lib.image_utils import augment


class NullSampleAugmenter:
    def augment(self, sample): return sample


class SampleAugmenter:
    def __init__(self, augment_threshold, translate_range_x=100, translate_range_y=10):
        self.augment_threshold = augment_threshold
        self.translate_range_x = translate_range_x
        self.translate_range_y = translate_range_y

    def augment(self, sample):
        if np.random.rand() >= self.augment_threshold:
            return sample

        center_image_path, left_image_path, right_image_path = sample.images()
        steering_angle = sample.steering_angle()

        return augment(
            center_image_path,
            left_image_path,
            right_image_path,
            steering_angle,
            self.translate_range_x,
            self.translate_range_y
        )
