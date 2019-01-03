import numpy as np

from lib.augmentation.data_sample import DataSample
from lib.augmentation.image_augmentation_utils import choose_image, \
    random_image_flip, \
    random_image_translate, \
    random_image_shadow, \
    random_image_brightness


class SampleAugmenter:

    @staticmethod
    def create_from(image_preprocessor, cfg):
        return SampleAugmenter(
            image_preprocessor,
            cfg['train.augment.threshold'],
            cfg['train.augment.translate_range_x'],
            cfg['train.augment.translate_range_y'],
            cfg['train.augment.choose_image_adjustment_angle'],
            cfg['train.augment.image_translate_angle_delta'],
            cfg['train.augment.throttle.steer_threshold'],
            cfg['train.augment.throttle.speed_threshold'],
            cfg['train.augment.throttle.delta']
        )

    def __init__(self, image_processor):
        self.image_processor = image_processor

    def augment(self, sample):
        raise NotImplementedError("Please Implement augment method!")


class NullSampleAugmenter(SampleAugmenter):
    def __init__(self, image_processor): super().__init__(image_processor)

    def augment(self, sample):
        image_path = sample.feature_image('center')
        image = self.image_processor.process(image_path)

        return DataSample(features=[image], labels=[sample.label('steering')])


class SampleAugmenter(NullSampleAugmenter):
    def __init__(
            self,
            image_processor,
            augment_threshold,
            translate_range_x=100,
            translate_range_y=10,
            choose_image_adjustment_angle=0.2,
            image_translate_angle_delta=0.002,
            steer_threshold=0.5,
            speed_threshold=20,
            throttle_delta=0.3
    ):
        super().__init__(image_processor)
        self.augment_threshold = augment_threshold
        self.translate_range_x = translate_range_x
        self.translate_range_y = translate_range_y
        self.choose_image_adjustment_angle = choose_image_adjustment_angle
        self.image_translate_angle_delta = image_translate_angle_delta
        self.steer_threshold = steer_threshold
        self.speed_threshold = speed_threshold
        self.throttle_delta = throttle_delta

    def augment(self, sample):
        if np.random.rand() > self.augment_threshold:
            return super().augment(sample)

        image, steering_angle = self.__augment_image_and_steering_angle(sample)

        return DataSample(
            features=[self.image_processor.process(image)],
            labels=[steering_angle]
        )

    def __augment_image_and_steering_angle(self, sample):
        image, steering_angle = choose_image(
            sample.feature('center'),
            sample.feature('left'),
            sample.feature('right'),
            sample.label('steering'),
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

    def __throttle_augmentation(self, sample, steer_threshold, speed_threshold, throttle_delta):
        throttle = sample.label('throttle')

        if np.random.rand() < 0.5 and \
                abs(sample.label('steering')) > steer_threshold and \
                sample.feature('speed') > speed_threshold:
            return throttle * throttle_delta

        return throttle
