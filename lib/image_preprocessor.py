from lib.utils.image_utils import vertical_crop_image, resize_image, rgb_to_yuv


class ImagePreprocessor:
    @staticmethod
    def create_from(config):
        return ImagePreprocessor(
            top_offset=config['train']['preprocess']['crop']['top_offset'],
            bottom_offset=config['train']['preprocess']['crop']['bottom_offset'],
            input_shape=(
                config['network']['image_input_shape']['height'],
                config['network']['image_input_shape']['width'],
                config['network']['image_input_shape']['channels']
            )
        )

    def __init__(self, top_offset, bottom_offset, input_shape):
        self.top_offset = top_offset
        self.bottom_offset = bottom_offset
        self.input_shape = input_shape

    def process(self, image):
        image = vertical_crop_image(image, self.top_offset, self.bottom_offset)
        image = resize_image(image, self.input_shape[1], self.input_shape[0])
        return rgb_to_yuv(image)
