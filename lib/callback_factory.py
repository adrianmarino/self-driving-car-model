from keras.callbacks import ModelCheckpoint


class CallbackFactory:
    @staticmethod
    def create_checkpoint_that_save_model_when_reach_better_metric_value(
            model_name='',
            metric='val_loss'
    ):
        return ModelCheckpoint(
            model_name + 'model-{epoch:03d}.h5',
            monitor=metric,
            verbose=1,
            save_best_only=True,
            mode='auto'
        )

 