import numpy as np


class NullSampleAugmenter:
    def augment(self, sample): return sample


class SampleAugmenter:
    def __init__(self, work_path, threshold):
        self.work_path = work_path
        self.threshold = threshold

    def augment(self, sample):
        if np.random.rand() >= self.threshold:
            return sample

        center_image, left_image, right_image = sample.images()
        steering_angle = sample.steering_angle()
