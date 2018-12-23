from keras.callbacks import LearningRateScheduler
from keras import backend as K


class LRScheduler(LearningRateScheduler):
    def __init__(self, epoch_lr, verbose=0):
        super().__init__(self.resolve, verbose)
        self.epoch_lr = epoch_lr

    def resolve(self, epoch):
        if epoch in self.epoch_lr:
            return self.epoch_lr[epoch]
        else:
            return K.eval(self.model.optimizer.lr)
