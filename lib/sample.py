from lib.image_utils import load_image
from lib.plot_utils import grid_display


class Sample:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def show(self):
        left, center, right = self.images()

        print("\n\n\nImages")
        grid_display(
            images=[left, center, right],
            titles=[f'Left Camera {left.shape}', f'Center Camera {center.shape}', f'Right Camera {left.shape}'],
            columns=3,
            figure_size=(40, 40),
            font_size=32
        )
        print(f'Steering Angle: {self.labels[0]}\n\n')

    def center_image_path(self): return self.features[0]

    def left_image_path(self): return self.features[1]

    def right_image_path(self): return self.features[2]

    def ordered_image_paths(self): return [self.left_image_path(), self.center_image_path(), self.right_image_path()]

    def images(self): return [load_image(path) for path in self.ordered_image_paths()]

    def center_image(self): return load_image(self.center_image_path())

    def left_image(self): return load_image(self.left_image_path())

    def right_image(self): return load_image(self.right_image_path())

    def steering_angle(self): return self.labels[0]
