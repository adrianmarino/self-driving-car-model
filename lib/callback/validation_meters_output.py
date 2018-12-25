import matplotlib.pyplot as plt
from IPython.display import clear_output
from keras.callbacks import Callback

from lib.model.metrics import MetricMeterBuilder


class ValidationMetersOutput(Callback):
    def __init__(self, validation_generator, evaluate_interval=50):
        super().__init__()
        self.evaluate_interval = evaluate_interval
        self.validation_generator = validation_generator
        self.val_bach_index = 0
        self.val_metrics_values = {}

    def on_train_begin(self, logs=None):
        for metric in self.model.metrics_names:
            self.val_metrics_values[metric] = []

    def on_batch_end(self, batch, logs=None):
        if batch % self.evaluate_interval == 0:
            val_features, val_labels = self.validation_generator[self.val_bach_index]
            score = self.model.evaluate(val_features, val_labels)

            index = 0
            output = []
            meter_builder = MetricMeterBuilder(self.val_metrics_values)
            for metric in self.model.metrics_names:
                self.val_metrics_values[metric].append(score[index])
                output.append(meter_builder.build(metric))
                index += 1

            print('Validation:')
            for line in output:
                print(f'- {line}')

        if self.val_bach_index < len(self.validation_generator):
            self.val_bach_index += 1
        else:
            self.val_bach_index = 0
