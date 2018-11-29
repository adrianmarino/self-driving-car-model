from sklearn.model_selection import train_test_split

from lib.plot_utils import show_example


class Example:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def show(self): show_example(self.features, self.labels)


class DataSet:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def split(self, percent):
        train_features, validation_features, \
         train_labels, validation_labels = train_test_split(
            self.features,
            self.labels,
            test_size=percent,
            random_state=0
         )

        return DataSet(train_features, train_labels), \
            DataSet(validation_features, validation_labels)

    def __getitem__(self, index):
        return Example(self.features[0], self.labels[0])

    def __len__(self): return len(self.features)
