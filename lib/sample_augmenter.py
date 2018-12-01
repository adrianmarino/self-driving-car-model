import numpy as np
from lib.image_augmentation_utils import choose_image, random_image_flip, random_image_translate, random_image_shadow, random_image_brightness


class NullSampleAugmenter:
    def augment(self, sample): return sample.center_image(), sample.steering_angle()


class SampleAugmenter:
    def __init__(self, augment_threshold, translate_range_x=100, translate_range_y=10):
        self.augment_threshold = augment_threshold
        self.translate_range_x = translate_range_x
        self.translate_range_y = translate_range_y

    def augment(self, sample):
        if np.random.rand() > self.augment_threshold:
            return sample.center_image(), sample.steering_angle()

        image, steering_angle = choose_image(
            sample.center_image_path(),
            sample.left_image_path(),
            sample.right_image_path(),
            sample.steering_angle()
        )

        image, steering_angle = random_image_flip(image, steering_angle)

        image, steering_angle = random_image_translate(
            image,
            steering_angle,
            self.translate_range_x,
            self.translate_range_y
        )

        image = random_image_shadow(image, image.shape[1], image.shape[0])

        image = random_image_brightness(image)

        return image, steering_angle
