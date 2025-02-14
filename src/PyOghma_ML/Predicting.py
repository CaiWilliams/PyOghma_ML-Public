import os
import tensorflow as tf


class Predicting:
    def __init__(self, network_dir, experimental_feature):
        self.network_dir = network_dir
        self.experimental_feature = experimental_feature.astype(float)
        self.load_network()

    def load_network(self):
        dir = os.path.join(self.network_dir, 'model.keras')
        self.model = tf.keras.models.load_model(dir)

    def predict(self):
        return self.model(self.experimental_feature)