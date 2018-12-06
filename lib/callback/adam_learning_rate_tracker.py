from keras.callbacks import Callback
from keras import backend as K


class AdamLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = K.eval(self.model.optimizer.lr)
        decay = K.eval(self.model.optimizer.decay)
        iterations = K.eval(self.model.optimizer.iterations)

        lr_with_decay = lr / (1. + decay * iterations)
        print(f'LR: {lr_with_decay}\n')
