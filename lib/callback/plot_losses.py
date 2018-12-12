import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output


class PlotLosses(keras.callbacks.Callback):
    def __init__(
            self,
            validation_generator,
            plot_interval=2,
            evaluate_interval=10
    ):
        super().__init__()
        self.plot_interval = plot_interval
        self.evaluate_interval = evaluate_interval
        self.validation_generator = validation_generator
        self.i = 0
        self.val_bach_index = 0
        self.x = []
        self.logs = []
        self.metrics_values = {}
        self.val_metrics_values = {}

    def on_train_begin(self, logs={}):
        for metric in self.model.metrics_names:
            self.metrics_values[metric] = []
            self.val_metrics_values[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.plot_interval == 0:
            clear_output(wait=True)
            f, axes = plt.subplots(1, len(self.model.metrics_names), sharex=True, figsize=(40, 8))

            index = 0
            for metric in self.model.metrics_names:
                axes[index].plot(self.x, self.metrics_values[metric], label=metric)
                axes[index].plot(self.x, self.val_metrics_values[metric], label=f'val_{metric}')
                axes[index].legend()
                index += 1

            plt.show()

    def on_batch_end(self, batch, logs={}):
        if batch % self.evaluate_interval == 0:
            self.i += 1
            self.logs.append(logs)
            self.x.append(self.i)

            val_features, val_labels = self.validation_generator[self.val_bach_index]
            score = self.model.evaluate(val_features, val_labels)

            index = 0
            output = []
            for metric in self.model.metrics_names:
                self.metrics_values[metric].append(logs.get(metric))
                self.val_metrics_values[metric].append(score[index])
                output.append(f'val_{metric}: {score[index]:.4f}')
                index += 1
            print(" - ".join(output) + "\n")

        if self.val_bach_index < len(self.validation_generator):
            self.val_bach_index += 1
        else:
            self.val_bach_index = 0






