import numpy as np
from lib.image_augmentation_utils import choose_image, random_image_flip, random_image_translate, random_image_shadow, random_image_brightness


class NullSampleAugmenter:
    def augment(self, sample):
        return sample.center_image(), sample.steering_angle(), sample.speed()*(1/31), sample.throttle()


class SampleAugmenter:
    def __init__(self,
                 augment_threshold,
                 translate_range_x=100,
                 translate_range_y=10,
                 choose_image_adjustment_angle=0.2,
                 image_translate_angle_delta=0.002
    ):
        self.augment_threshold = augment_threshold
        self.translate_range_x = translate_range_x
        self.translate_range_y = translate_range_y
        self.choose_image_adjustment_angle = choose_image_adjustment_angle
        self.image_translate_angle_delta = image_translate_angle_delta

    def augment(self, sample):
        if np.random.rand() > self.augment_threshold:
            return NullSampleAugmenter().augment(sample)

        image, steering_angle = self.augment_image_and_steering_angle(sample)

        return image, steering_angle, sample.speed()*(1/31), sample.throttle()

    def augment_image_and_steering_angle(self, sample):
        image, steering_angle = choose_image(
            sample.center_image_path(),
            sample.left_image_path(),
            sample.right_image_path(),
            sample.steering_angle(),
            self.choose_image_adjustment_angle
        )
        image, steering_angle = random_image_flip(image, steering_angle)
        image, steering_angle = random_image_translate(
            image,
            steering_angle,
            self.translate_range_x,
            self.translate_range_y,
            self.image_translate_angle_delta
        )
        image = random_image_shadow(image, image.shape[1], image.shape[0])
        image = random_image_brightness(image)
        return image, steering_angle

