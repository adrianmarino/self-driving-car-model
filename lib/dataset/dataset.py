import numpy as np
from sklearn.model_selection import train_test_split

from lib.plot_utils import histograms
from lib.dataset.dataset_sample import DatasetSample


class Dataset:
    def __init__(self, features, feature_columns, labels, label_columns, shuffle=False):
        self.features = features
        self.labels = labels
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.indexes = np.arange(len(features))
        if shuffle:
            np.random.shuffle(self.indexes)

    def split(self, percent, shuffle=True):
        train_features, validation_features, train_labels, validation_labels = train_test_split(
            self.features,
            self.labels,
            test_size=percent,
            random_state=0,
            shuffle=shuffle
        )
        return Dataset(train_features, self.feature_columns, train_labels, self.label_columns), \
            Dataset(validation_features, self.feature_columns, validation_labels, self.label_columns)

    def __getitem__(self, index):
        features = self.features[self.indexes[index]]
        labels = self.labels[self.indexes[index]]
        return DatasetSample(features, self.feature_columns, labels, self.label_columns)

    def __len__(self): return len(self.features)

    def shuffle(self):
        return Dataset(
            self.features,
            self.feature_columns,
            self.labels,
            self.label_columns,
            shuffle=True
        )

    def subset(self, index, size):
        initial_position, final_position = index * size, (index + 1) * size
        return Dataset(
            self.features[initial_position:final_position],
            self.feature_columns,
            self.labels[initial_position:final_position],
            self.label_columns
        )

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self):
            sample = self[self.index]
            self.index += 1
            return sample
        else:
            raise StopIteration

    def label_column(self, index): return self.labels[:, index]

    def show_labels_stats(self):
        histograms(
            values=[self.label_column(index) for index in range(len(self.label_columns))],
            x_labels=[col.capitalize() for col in self.label_columns],
            titles=[f'Range: ({self.label_column(i).min():0.4f} , {self.label_column(i).max():0.4f})' for i in range(len(self.label_columns))]
        )
