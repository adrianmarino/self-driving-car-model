import os

from keras.callbacks import ModelCheckpoint
from lib.callback.plot_losses import PlotLosses
from lib.os_utils import create_directory


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

    def checkpoint_file_path(self):
        return os.path.join(self.path, f'{self.checkpoint_filename()}.h5')

    @staticmethod
    def checkpoint_filename():
        filename = f'model_weights-'
        filename += 'epoch_{epoch:03d}-'
        filename += 'steer_rmse_{val_steer_rmse:.4f}-'
        filename += 'throttle_rmse_{val_throttle_rmse:.4f}'
        return filename


class PlotLossesFactory:
    @staticmethod
    def create(validation_generator, plot_interval=1, evaluate_interval=10):
        return PlotLosses(validation_generator, plot_interval, evaluate_interval)
