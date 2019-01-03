import argparse
import warnings

# Model
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

from lib.augmentation.sample_augmenter import SampleAugmenter, NullSampleAugmenter
from lib.callback.adam_learning_rate_tracker import AdamLearningRateTracker
# Callbacks
from lib.callback.checkpoint import CheckpointFactory
from lib.callback.validation_meters_output import ValidationMetersOutput
from lib.config import Config
from lib.data_generator import DataGenerator
# Data generation & augmentation
from lib.dataset.dataset_loader import DatasetLoader
from lib.image_preprocessor import ImagePreprocessor
from lib.model.metrics import rmse
from lib.model.model_factory import ModelFactory
from lib.utils.file_utils import last_created_file_from


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def create_data_generators(cfg):
    loader = DatasetLoader(cfg)
    data_set = loader.load(features=cfg['dataset.features'], labels=cfg['dataset.labels'])
    train_set, validation_set = data_set.split(percent=cfg['train.validation_set_percent'], shuffle=True)

    image_preprocessor = ImagePreprocessor.create_from(cfg)

    train_augmenter = SampleAugmenter.create_from(image_preprocessor, cfg)
    train_generator = DataGenerator.create_from(train_set, train_augmenter, cfg)

    validation_augmenter = NullSampleAugmenter(image_preprocessor)
    validation_generator = DataGenerator.create_from(validation_set, validation_augmenter, cfg, shuffle_per_epoch=False)

    return train_set, train_generator, validation_generator


def try_to_load_weights(model):
    try:
        last_weights_file_path = last_created_file_from("checkpoints/*.h5")
        print(f'last_weights_file_path: {last_weights_file_path}')
        model.load_weights(last_weights_file_path)
    except:
        print("Not found weights file")


def build_callbacks(cfg, monitor_metric='val_rmse', evaluate_interval=100):
    return [
        CheckpointFactory(path=cfg['train.checkpoint_path']).create(metric=monitor_metric),
        EarlyStopping(monitor=monitor_metric, patience=4),
        ValidationMetersOutput(validation_generator, evaluate_interval),
        AdamLearningRateTracker(evaluate_interval),
        TensorBoard(
            log_dir='./logs',
            histogram_freq=0,
            write_graph=True,
            write_images=True,
            batch_size=cfg['train.batch_size'],
            write_grads=True,
            update_freq='batch'
        )
    ]


def params(cfg):
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument(
        '--epochs',
        help='Number of epochs to train.',
        type=int,
        default=int(cfg['train.epochs'])
    )
    parser.add_argument(
        '--lr',
        help='Learning rate.',
        type=float,
        default=float(cfg['train.lr'])
    )
    return parser.parse_args()


def create_model(params):
    model = ModelFactory.create_nvidia_model(loss='mean_squared_error', metrics=[rmse], optimizer=Adam(lr=params.lr))
    try_to_load_weights(model)
    return model


if __name__ == '__main__':
    cfg = Config('./config.yml')
    params = params(cfg)

    train_set, train_generator, validation_generator = create_data_generators(cfg)
    model = create_model(params)
    steps_per_epoch = int(len(train_set) / cfg['train.batch_size'])

    model.train(
        train_generator,
        validation_generator,
        steps_per_epoch,
        epochs=params.epochs,
        callbacks=build_callbacks(cfg)
    )
