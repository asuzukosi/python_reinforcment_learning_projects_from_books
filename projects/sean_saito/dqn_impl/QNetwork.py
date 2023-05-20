import tensorflow as tf
import numpy as np
import keras
import warnings

# self.net["conv1"] = tf.compat.v1.layers.conv2d(self.net["input"], 32, kernel=(8,8), stride=(4, 4), init_b=init_b, name="conv1")
# self.net["conv2"] = tf.compat.v1.layers.conv2d(self.net["conv1"], 64, kernel=(4,4), stride=(2, 2), init_b=init_b, name="conv2")
# self.net["conv3"] = tf.compat.v1.layers.conv2d(self.net["conv2"], 64, kernel=(3,3), stride=(1, 1), init_b=init_b, name="conv3")
# self.net["feature"] = tf.compat.v1.layers.dense(self.net["conv3"], 512, init_b=init_b, name="fc1")

class QNetworkBuilder:
    def __init__(self, input_shape, n_outputs, checkpoint_path):
        # the shape of the input data coming into the network
        self.input_shape = input_shape
        # spcifiy the number of outputs ie the number of actions
        self.n_outputs = n_outputs
    
    def build(self):
        # create sequential convolutional network
        if self.has_model():
            warnings.warn("QNetwork already exists, this operation would be creating a new QNetwork")
        
        self.model = keras.models.Sequential(
            tf.keras.layers.Conv2D(filters=32, kernel_size=(8, 8), activation="relu",strides=(4, 4), name="conv1"),
            tf.keras.layers.Dropout(rate=0.2, name="drop1"),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), activation="relu",strides=(2, 2), name="conv2"),
            tf.keras.layers.Dropout(rate=0.2, name="drop1"),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), name="conv3"),
            tf.keras.layers.BatchNormalization(name="batchnorm"),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(512, activation="relu", name="fc1"),
            tf.keras.layers.Dense(self.n_outputs, name="values"),
        )
        
        # build the model
        self.model.build(input_shape=self.input_shape)
        print(self.model.summary())
        return self.model
    
    def has_model(self):
        # checks if the model exists
        if self.model is not None:
            return True
        return False
    
    def predict(self, X):
        pass
    
    
        