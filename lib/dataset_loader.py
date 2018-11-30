import pandas as pd
import os
from lib.dataset import Dataset


class DatasetLoader:
    def __init__(self, config): self.__config = config

    def __path(self): return os.path.join(os.getcwd(), self.__config['dataset']['path'])

    def __columns(self): return self.__config['dataset']['columns']

    def load(self, features, labels):
        data_frame = pd.read_csv(self.__path(), names=self.__columns())
        return Dataset(data_frame[features].values, data_frame[labels].values)
