import os

from keras.callbacks import ModelCheckpoint
from lib.callback.plot_losses import PlotLosses
from lib.os_utils import create_directory


class CheckpointFactory:
    def __init__(self, path):
        create_directory(path)
        self.path = path

    def create(
            self,
            model_name='',
            metric='val_loss'
    ):
        return ModelCheckpoint(
            self.checkpoint_file_path(model_name),
            monitor=metric,
            verbose=1,
            save_best_only=True,
            mode='auto'
        )

    def checkpoint_file_path(self, model_name):
        return os.path.join(self.path, f'{self.checkpoint_filename(model_name)}.h5')

    @staticmethod
    def checkpoint_filename(model_name):
        filename = f'{model_name.lower()}_mode_weights--'
        filename += 'epoch_{epoch:03d}--'
        filename += 'val_loss_{val_loss:.4f}--'
        filename += 'val_acc_{val_acc:.4f}--'
        filename += 'val_loss_{loss:.4f}--'
        filename += 'val_acc_{acc:.4f}'
        return filename


class PlotLossesFactory:
    @staticmethod
    def create(
            plot_interval=1,
            evaluate_interval=10,
            x_val=None,
            y_val_categorical=None
    ):
        return PlotLosses(plot_interval, evaluate_interval, x_val, y_val_categorical)
