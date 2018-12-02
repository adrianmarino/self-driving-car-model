import numpy as np
from sklearn.model_selection import train_test_split

from lib.sample import Sample


class Dataset:
    def __init__(self, features, labels, shuffle=False):
        self.features = features
        self.labels = labels
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
        return Dataset(train_features, train_labels), Dataset(validation_features, validation_labels)

    def __getitem__(self, index):
        features = self.features[self.indexes[index]]
        labels = self.labels[self.indexes[index]]
        return Sample(features, labels)

    def __len__(self): return len(self.features)

    def shuffle(self): return Dataset(self.features, self.labels, shuffle=True)

    def subset(self, index, size):
        initial_position, final_position = index * size, (index + 1) * size
        return Dataset(
            self.features[initial_position:final_position],
            self.labels[initial_position:final_position]
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
