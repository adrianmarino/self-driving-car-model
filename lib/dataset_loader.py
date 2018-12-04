import pandas as pd
import os
from lib.dataset import Dataset


class DatasetLoader:
    def __init__(self, config): self.__config = config

    def __paths(self): return [os.path.join(os.getcwd(), path) for path in self.__config['dataset']['paths']]

    def __columns(self): return self.__config['dataset']['columns']

    def load(self, features, labels):
        data_frames = []
        for path in self.__paths():
            dataset = pd.read_csv(path, names=self.__columns())
            print(f'dataset({len(dataset)}) loadded {path} ')
            data_frames.append(dataset)

        data_frame = pd.concat(data_frames)

        return Dataset(data_frame[features].values, data_frame[labels].values)
