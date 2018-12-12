import pandas as pd
import os
from lib.dataset.dataset import Dataset


class DatasetLoader:
    def __init__(self, config): self.__config = config

    def __paths(self, mode):
        return [os.path.join(os.getcwd(), path) for path in self.__config['dataset']['paths'][mode]]

    def __columns(self): return self.__config['dataset']['columns']

    def load(self, features, labels, mode='train'):
        data_frames = []
        for path in self.__paths(mode):
            dataset = pd.read_csv(path, names=self.__columns())
            print(f'dataset({len(dataset)}) loadded {path} ')
            data_frames.append(dataset)

        data_frame = pd.concat(data_frames)

        return Dataset(
            data_frame[features].values,
            features,
            data_frame[labels].values,
            labels
        )
