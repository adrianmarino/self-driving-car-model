import os
from keras.callbacks import ModelCheckpoint
from lib.utils.os_utils import create_directory


class CheckpointFactory:
    def __init__(self, path):
        create_directory(path)
        self.path = path

    def create(self, metric='val_loss'):
        return ModelCheckpoint(
            self.checkpoint_file_path(),
            monitor=metric,
            verbose=1,
            save_best_only=True,
            mode='auto'
        )

    def checkpoint_file_path(self): return os.path.join(self.path, f'{self.checkpoint_filename()}.h5')

    @staticmethod
    def checkpoint_filename(): return 'weights__epoch_{epoch:02d}__loss_{loss:.4f}__rmse_{val_rmse:.4f}'
