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

cfg = Config('./config.yml')

image_input_shape = (
    cfg['network']['image_input_shape']['height'],
    cfg['network']['image_input_shape']['width'],
    cfg['network']['image_input_shape']['channels']
)
input_shapes = [image_input_shape]
output_shape = (1,)

batch_size = cfg['train']['batch_size']
augment_threshold = cfg['train']['augment']['threshold']
translate_range_x = cfg['train']['augment']['translate_range_x']
translate_range_y = cfg['train']['augment']['translate_range_y']
top_offset = cfg['train']['preprocess']['crop']['top_offset']
bottom_offset = cfg['train']['preprocess']['crop']['bottom_offset']
steer_threshold = cfg['train']['augment']['throttle']['steer_threshold']
speed_threshold = cfg['train']['augment']['throttle']['speed_threshold']
throttle_delta = cfg['train']['augment']['throttle']['throttle_delta']
choose_image_adjustment_angle = cfg['train']['augment']['choose_image_adjustment_angle']
image_translate_angle_delta = cfg['train']['augment']['image_translate_angle_delta']

loader = DatasetLoader(cfg)
data_set = loader.load(features=cfg['dataset']['features'], labels=cfg['dataset']['labels'])
train_set, validation_set = data_set.split(percent=cfg['train']['validation_set_percent'], shuffle=True)

image_preprocessor = ImagePreprocessor(top_offset, bottom_offset, image_input_shape)

train_augmenter = SampleAugmenter(
    image_preprocessor,
    augment_threshold,
    translate_range_x,
    translate_range_y,
    choose_image_adjustment_angle,
    image_translate_angle_delta,
    steer_threshold,
    speed_threshold,
    throttle_delta
)

train_generator = DataGenerator(
    train_set,
    input_shapes,
    output_shape,
    batch_size,
    train_augmenter,
    shuffle_per_epoch=True
)

validation_augmenter = NullSampleAugmenter(image_preprocessor)

validation_generator = DataGenerator(
    validation_set,
    input_shapes,
    output_shape,
    batch_size,
    validation_augmenter,
    shuffle_per_epoch=False
)

model = ModelFactory.create_nvidia_model(
    loss='mean_squared_error',
    metrics=[rmse],
    # optimizer=Adam(lr=0.001)
    optimizer = Adam(lr=0.0001)
    # optimizer = Adam(lr=0.00001)
)

try:
    last_weights_file_path = last_created_file_from("checkpoints/*.h5")
    print(f'last_weights_file_path: {last_weights_file_path}')
    model.load_weights(last_weights_file_path)
except:
    print("Not found weights file")

steps_per_epoch = int(len(train_set) / batch_size)
epochs = 25
monitor_metric = 'val_rmse'
evaluate_interval = 100

callbacks = [
    CheckpointFactory(path=cfg['train']['checkpoint_path']).create(metric=monitor_metric),
    EarlyStopping(monitor=monitor_metric, patience=4),
    ValidationMetersOutput(validation_generator, evaluate_interval),
    AdamLearningRateTracker(evaluate_interval),
    TensorBoard(
        log_dir='./logs',
        histogram_freq=0,
        write_graph=True,
        write_images=True,
        batch_size=batch_size,
        write_grads=True,
        update_freq='batch'
    )
]

model.train(train_generator, validation_generator, steps_per_epoch, epochs, callbacks)
