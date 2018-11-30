from keras.optimizers import Adam

from lib.plot_utils import graph_model, show_img


class Model:
    def __init__(self, model, model_name, description_image_path=None):
        self.model = model
        self.name = model_name
        self.description_image_path = description_image_path

    def compile(
            self,
            loss='mean_squared_error',
            optimizer=Adam(lr=1.0e-4)
    ):
        self.model.compile(loss=loss, optimizer=optimizer)

    def train(
            self,
            train_generator,
            validation_generator,
            steps_per_epoch=20000,
            epochs=1,
            max_q_size=1,
            validation_samples_size=1,
            use_multiprocessing=True,
            workers=8,
            callbacks=()
    ):
        return self.model.fit_generator(
            train_generator,
            steps_per_epoch,
            epochs,
            max_q_size=max_q_size,
            validation_data=validation_generator,
            nb_val_samples=validation_samples_size,
            callbacks=callbacks,
            verbose=1,
            use_multiprocessing=use_multiprocessing,
            workers=workers
        )

    def show(self):
        print("\n\n\nMODEL LAYERS\n")
        self.model.summary()
        print("\n\n\nMODEL GRAPH\n")
        if self.description_image_path is not None:
            show_img(self.description_image_path, size=(12, 12))
        graph_model(self.model)
