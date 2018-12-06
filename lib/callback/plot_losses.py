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
        self.val_batch_count = len(self.validation_generator)
        self.val_bach_index = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.logs = []

    def on_train_begin(self, logs={}):
        print('Begin training')

    def on_epoch_end(self, epoch, logs={}):
        if self.evaluate_interval is None:
            self.logs.append(logs)
            self.x.append(self.i)
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.acc.append(logs.get('acc'))
            self.val_acc.append(logs.get('val_acc'))
            self.i += 1

        if epoch % self.plot_interval == 0:
            clear_output(wait=True)
            f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(20, 5))
            ax1.plot(self.x, self.losses, label="loss")
            ax1.plot(self.x, self.val_losses, label="val_loss")
            ax1.legend()

            ax2.plot(self.x, self.acc, label="acc")
            ax2.plot(self.x, self.val_acc, label="val_acc")
            ax2.legend()
            plt.show()

    def on_batch_end(self, batch, logs={}):
        if self.evaluate_interval is not None and batch % self.evaluate_interval == 0:
            self.i += 1
            self.logs.append(logs)
            self.x.append(self.i)
            self.losses.append(logs.get('loss'))
            self.acc.append(logs.get('acc'))

            if self.validation_generator is not None:
                val_batch = self.validation_generator[self.val_bach_index]
                score = self.model.evaluate(val_batch[0], val_batch[1], verbose=0)
                self.val_losses.append(score[0])
                self.val_acc.append(score[1])
                print(f' - val_loss: {score[0]:.4f} - val_acc: {score[1]:.4f}')

        self.increment_batch_index()

    def increment_batch_index(self):
        if self.val_bach_index == self.val_batch_count:
            self.val_bach_index = 0
        else:
            self.val_bach_index += 1



