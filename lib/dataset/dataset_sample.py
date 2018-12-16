from lib.image_utils import load_image


class DatasetSample:
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

    def show_features(self, feature_columns=[]):
        if len(feature_columns) == 0:
            feature_columns = self.feature_columns

        print("Features")
        for feature_name in feature_columns:
            print(f'\t- {feature_name.capitalize()}: {self.feature(feature_name)}')

    def feature_image(self, name): return load_image(self.feature(name))

    def label(self, name):
        index = self.labels_columns.index(name)
        return self.labels[index]

    def feature(self, name):
        index = self.feature_columns.index(name)
        return self.features[index]
