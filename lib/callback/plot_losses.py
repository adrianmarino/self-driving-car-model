import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output


class PlotLosses(keras.callbacks.Callback):
    def __init__(
            self,
            validation_generator,
            plot_interval=2,
            evaluate_interval=10,
            metric=None
    ):
        super().__init__()
        self.plot_interval = plot_interval
        self.evaluate_interval = evaluate_interval
        self.validation_generator = validation_generator
        self.metric = metric
        self.i = 0
        self.val_bach_index = 0
        self.x = []

        self.loss_values = []
        self.val_loss_values = []

        self.metric_values = []
        self.val_metric_values = []

        self.logs = []

    def on_train_begin(self, logs={}):
        print('Begin training')

    def on_epoch_end(self, epoch, logs={}):
        if self.evaluate_interval is None:
            self.logs.append(logs)
            self.x.append(self.i)

            self.loss_values.append(logs.get('loss'))
            self.val_loss_values.append(logs.get('val_loss'))

            if self.metric is not None:
                self.metric_values.append(logs.get(self.metric))
                self.val_metric_values.append(logs.get(f'val_{self.metric}'))

            self.i += 1

        if epoch % self.plot_interval == 0:
            clear_output(wait=True)
            f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(20, 5))

            ax1.plot(self.x, self.loss_values, label="loss")
            ax1.plot(self.x, self.val_loss_values, label="val_loss")
            ax1.legend()

            if self.metric is not None:
                ax2.plot(self.x, self.metric_values, label=self.metric)
                ax2.plot(self.x, self.val_metric_values, label=f'val_{self.metric}')
                ax2.legend()

            plt.show()

    def on_batch_end(self, batch, logs={}):
        if self.evaluate_interval is not None and batch % self.evaluate_interval == 0:
            self.i += 1
            self.logs.append(logs)
            self.x.append(self.i)
            self.loss_values.append(logs.get('loss'))

            if self.metric is not None:
                self.metric_values.append(logs.get(self.metric))

            if self.validation_generator is not None:
                val_batch = self.validation_generator[self.val_bach_index]
                score = self.model.evaluate(val_batch[0], val_batch[1], verbose=0)

                self.val_loss_values.append(score[0])

                if self.metric is not None:
                    self.val_metric_values.append(score[1])

                if self.metric is not None:
                    print(f' - val_loss: {score[0]:.4f} - val_{self.metric}: {score[1]:.4f}')
                else:
                    print(f' - val_loss: {score[0]:.4f}')

        if self.val_bach_index < len(self.validation_generator):
            self.val_bach_index += 1
        else:
            self.val_bach_index = 0




