import cv2
import matplotlib.image as img


def load_image(path): return img.imread(path.strip())


def vertical_crop_image(image, top_offset, bottom_offset):
    """
    Remove a top & button amount of positions. i.e. given an (100, 100, 3) image and applies:

    vertical_crop(image, 30, 30)

    That remove first 60 and last 60 positions from Y axis. Then the results is (40, 100, 3)
    """
    return image[top_offset:-bottom_offset, :, :]


def resize_image(image, width, height):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (width, height), cv2.INTER_AREA)


def rgb_to_yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
