from lib.utils.image_utils import load_image


class DatasetSample:
    def __init__(self, features, feature_columns, labels, labels_columns):
        self.features = features
        self.labels = labels
        self.feature_columns = feature_columns
        self.labels_columns = labels_columns

    def show(self):
        print(self.str_features())
        print(self.str_labels())

    def str_labels(self):
        labels = []
        for index, label in enumerate(zip(self.labels_columns, self.labels)):
            labels.append(f'\t- {label[0].capitalize()}: {label[1]}')
        return 'Labels\n' + '\n'.join(labels)

    def str_features(self, feature_columns=[]):
        if len(feature_columns) == 0:
            feature_columns = self.feature_columns

        features = []
        for feature_name in feature_columns:
            features.append(f'\t- {feature_name.capitalize()}: {self.feature(feature_name)}')

        return 'Features\n' + '\n'.join(features)

    def feature_image(self, name): return load_image(self.feature(name))

    def label(self, name):
        index = self.labels_columns.index(name)
        return self.labels[index]

    def feature(self, name):
        index = self.feature_columns.index(name)
        return self.features[index]
    
    def __str__(self): return f'{self.str_features()}\n{self.str_labels()}'
