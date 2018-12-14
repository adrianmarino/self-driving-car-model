import numpy as np

from lib.augmentation.data_sample import DataSample
from lib.augmentation.image_augmentation_utils import choose_image, \
    random_image_flip, \
    random_image_translate, \
    random_image_shadow, \
    random_image_brightness


class SampleAugmenter:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def augment(self, sample):
        raise NotImplementedError("Please Implement augment method!")


class NullSampleAugmenter(SampleAugmenter):
    def __init__(self, image_processor): super().__init__(image_processor)

    def augment(self, sample):
        return DataSample(
            self.image_processor.process(sample.center_image()),
            [sample.steering_angle(), sample.throttle()]
        )


class SampleAugmenter(NullSampleAugmenter):
    def __init__(
            self,
            image_processor,
            augment_threshold,
            translate_range_x=100,
            translate_range_y=10,
            choose_image_adjustment_angle=0.2,
            image_translate_angle_delta=0.002
    ):
        super().__init__(image_processor)
        self.augment_threshold = augment_threshold
        self.translate_range_x = translate_range_x
        self.translate_range_y = translate_range_y
        self.choose_image_adjustment_angle = choose_image_adjustment_angle
        self.image_translate_angle_delta = image_translate_angle_delta

    def augment(self, sample):
        if np.random.rand() > self.augment_threshold:
            return super().augment(sample)

        image, steering_angle = self.__augment_image_and_steering_angle(sample)

        return DataSample(
            self.image_processor.process(image),
            [steering_angle, sample.throttle()]
        )

    def __augment_image_and_steering_angle(self, sample):
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
