import numpy as np
# import tensorflow as tf
import sklearn.preprocessing as prep
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


def min_max_scale(X_train, X_test):
    preprocessor = prep.MinMaxScaler().fit(np.concatenate((X_train, X_test), axis=0))
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def mean_normalization(X_train, X_test):
    data = np.concatenate((X_train, X_test), axis=0)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return (X_train - mean) / std, (X_test - mean) / std


def xavier_init(fan_in, fan_out, function):
    if function is tf.nn.sigmoid:
        low = -4.0 * np.sqrt(6.0 / (fan_in + fan_out))
        high = 4.0 * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
    elif function is tf.nn.tanh:
        low = -1 * np.sqrt(6.0 / (fan_in + fan_out))
        high = 1 * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
    else:
      low = -4.0 * np.sqrt(6.0 / (fan_in + fan_out)) * 1e-3
      high = 4.0 * np.sqrt(6.0 / (fan_in + fan_out)) * 1e-3
      print('>>>>>>>>>>>>>>>>>>>> low and high for initializer: ',low,high)
      # np.random.normal(-1e-2, 1e-2, [self.n_input, self.n_hidden])
      return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
