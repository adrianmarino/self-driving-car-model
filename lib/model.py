from keras.optimizers import Adam

from lib.image_utils import load_image
from lib.plot_utils import graph_model, show_image


class Model:
    def __init__(self, model, model_name, description_image_path=None):
        self.model = model
        self.name = model_name
        self.description_image_path = description_image_path

    def train(
            self,
            generator,
            validation_generator,
            steps_per_epoch=20000,
            epochs=10,
            callbacks=[]
    ):
        return self.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1,
            use_multiprocessing=True,
            workers=8
        )

    def show(self):
        print("\n\n\nMODEL LAYERS\n")
        self.model.summary()
        print("\n\n\nMODEL GRAPH\n")
        if self.description_image_path is not None:
            show_image(load_image(self.description_image_path), size=(12, 12))
        graph_model(self.model)

    def evaluate(self, features, labels):
        return self.model.evaluate(features, labels, verbose=0)

    def load_weights(self, path='model.h5'): self.model.load_weights(path)