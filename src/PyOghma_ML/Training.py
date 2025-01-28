import os
import numpy as np
import tensorflow as tf
from dataclasses import dataclass

class Training:
    def __init__(self, training_features, training_targets, validation_features, validation_targets, dir, model_settings=None):
        self.model = tf.keras.models.Sequential()
        if model_settings is not None:
            self.model_settings = model_settings
        else:
            self.model_settings = Model_Settings()
        self.input_dim = len(training_features[0])
        self.output_dim = np.shape(training_targets)[1]
        print(self.output_dim)
        self.validation_features = validation_features.astype(float)
        self.validation_targets = validation_targets.astype(float)
        self.setup_model()
        self.fitting(training_features, training_targets, validation_features, validation_targets)
        self.saveing(dir)


    def setup_model(self):
        for i, nodes in enumerate(self.model_settings.layer_nodes):
            if i == 0:
                self.model.add(tf.keras.layers.Dense(nodes, kernel_initializer=self.model_settings.initializer, kernel_regularizer=self.model_settings.regularization))
                self.model.add(tf.keras.layers.Activation(self.model_settings.activation))
                if self.model_settings.batch_norm:
                    self.model.add(tf.keras.layers.BatchNormalization())
                if len(self.model_settings.dropout) != 0:
                    self.model.add(tf.keras.layers.Dropout(self.model_settings.dropout[i]))
            else:
                self.model.add(tf.keras.layers.Dense(nodes, kernel_initializer=self.model_settings.initializer, kernel_regularizer=self.model_settings.regularization))
                self.model.add(tf.keras.layers.Activation(self.model_settings.activation))
                if self.model_settings.batch_norm:
                    self.model.add(tf.keras.layers.BatchNormalization())
                if len(self.model_settings.dropout) != 0:
                    self.model.add(tf.keras.layers.Dropout(self.model_settings.dropout[i]))

        self.model.add(tf.keras.layers.Dense(self.output_dim, activation=None))

        model_optimiser = tf.keras.optimizers.Adam(learning_rate=self.model_settings.learning_rate)
        self.model.compile(optimizer=model_optimiser, loss=self.model_settings.loss_function, metrics=self.model_settings.metrics)

    def fitting(self, x, y, val_x, val_y):
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.model_settings.patience, verbose=1)]
        history = self.model.fit(x, y, batch_size=self.model_settings.batch_size, epochs=self.model_settings.epochs, callbacks=callbacks, shuffle=True, validation_data=(val_x, val_y))
        quality = self.model.evaluate(val_x, val_y, verbose=0)
        return history.history['val_mean_absolute_error'][-1]

    def saveing(self, dir):
        dir = os.path.join(dir, 'model.keras')
        self.model.save(dir)


class lr(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, gamma, power):
        self.initial_learning_rate = initial_learning_rate
        self.gamma = gamma
        self.power = power

    def __call__(self, step):
        return self.initial_learning_rate * tf.pow((step * self.gamma +1), -self.power)

    def get_config(self):
        config = {
            'initial_learning_rate': self.initial_learning_rate,
            'gamma': self.gamma,
            'power': self.power
        }
        return config

@dataclass
class Model_Settings:
    initializer = 'glorot_uniform'
    activation = 'gelu'
    regularization = None
    layer_nodes = [2048, 2048, 2048, 2048]
    dropout = [0.01, 0.01, 0.01, 0.01]
    batch_norm = True
    epochs = 256
    learning_rate = lr(1e-2, 0.001, 3)
    batch_size = 2024
    patience = 16
    loss_function = 'mse'
    metrics = [tf.keras.metrics.MeanAbsoluteError()]
    training_percentage = 0.8
    validation_percentage = 0.2
    permutations_limit = 4000
    ensemble_presample = 1
    ensemble_maximum = 2
    ensemble_tollerance = 1e-3
    ensemble_patience = 10



