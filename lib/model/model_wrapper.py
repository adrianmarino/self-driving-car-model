from lib.utils.plot_utils import graph_model, show_values


class ModelWrapper:
    def __init__(self, model, model_name):
        self.model = model
        self.name = model_name

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
        graph_model(self.model)

    def evaluate(self, features, labels, verbose=0):
        score = self.model.evaluate(features, labels, verbose=verbose)
        if verbose:
            show_values(self.metrics_names(), score)
        return score

    def load_weights(self, path='model.h5'):
        self.model.load_weights(path)

    def predict(self, x, batch_size=1):
        return self.model.predict(x, batch_size=batch_size)

    def metrics_names(self): return self.model.metrics_names
