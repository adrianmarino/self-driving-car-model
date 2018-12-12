from lib.image_utils import load_image
from lib.plot_utils import grid_display


class Sample:
    def __init__(self, features, feature_columns, labels, labels_columns):
        self.features = features
        self.labels = labels
        self.feature_columns = feature_columns
        self.labels_columns = labels_columns

    def show(self):
        self.show_features()
        self.show_labels()

    def show_labels(self):
        print("Labels")
        for index, label in enumerate(zip(self.labels_columns, self.labels)):
            print(f'\t- {label[0].capitalize()}: {label[1]}')

    def show_features(self):
        print("\n\n\nFeatures")
        images = self.images()
        titles = [f'{label} Camera {images[0].shape}' for label in self.feature_columns]
        grid_display(
            images=images,
            titles=titles,
            columns=len(titles),
            figure_size=(40, 40),
            font_size=22
        )

    def center_image_path(self): return self.features[0]

    def left_image_path(self): return self.features[1]

    def right_image_path(self): return self.features[2]

    def ordered_image_paths(self): return [self.left_image_path(), self.center_image_path(), self.right_image_path()]

    def images(self): return [load_image(path) for path in self.ordered_image_paths()]

    def center_image(self): return load_image(self.center_image_path())

    def left_image(self): return load_image(self.left_image_path())

    def right_image(self): return load_image(self.right_image_path())

    def steering_angle(self): return self.labels[0]

    def speed(self): return self.labels[1]

    def throttle(self): return self.labels[2]
