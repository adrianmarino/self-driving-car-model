import keras.backend as K


def rmse(y_true, y_pred): return root_mean_squared_error(y_true, y_pred)


def root_mean_squared_error(y_true, y_pred): return K.sqrt(K.mean(K.square(y_pred - y_true)))
