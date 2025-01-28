import copy
import os
import itertools
import random

import numpy as np
import pandas as pd
import ujson as json
import matplotlib.pyplot as plt

from scipy import interpolate
from scipy import stats

from .Predicting import Predicting
from .Training import Training, Model_Settings
from .Labels import Label


# TODO intergrate Oghma names into code

class Networks:
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def initialise(cls, networks_dir, network_type=None, model_settings=None):
        if network_type not in cls.subclasses:
            raise ValueError('Network Type: {} Not recognized'.format(network_type))
        return cls.subclasses[network_type](networks_dir, model_settings=model_settings)

    def setup_network_directories(self):
        oghma_network_config = os.path.join(self.networks_dir, 'faster', 'nets.json')

        if os.path.isfile(oghma_network_config) == False:
            raise ValueError('Network Config File Not Found')
        else:
            f = open(oghma_network_config, 'r')
            oghma_network_config = json.load(f)
            f.close()

        self.networks_configured = list(oghma_network_config['sims'].keys())
        self.working_network = 0
        self.total_networks = len(self.networks_configured)
        self.networks = np.zeros(len(self.networks_configured), dtype=object)

        self.oghma_network_config = oghma_network_config


        for network in self.networks_configured:
            network_dir = os.path.join(self.networks_dir, 'faster', network)
            if os.path.isdir(network_dir) == False:
                os.mkdir(network_dir)

    def load_input_vectors(self):
        input_vectors = {}
        input_experiments = self.oghma_network_config['experimental']
        self.networks_configured = list(self.oghma_network_config['sims'].keys())
        for experiment in input_experiments:
            vector = self.oghma_network_config['experimental'][experiment]['vec']['points'].split(',')
            vector = np.asarray(vector).astype(float)
            input_vectors[experiment] = vector
        self.input_vectors = input_vectors
        self.points = len(vector)

    def load_training_dataset(self):
        training_dataset = pd.read_csv(self.oghma_network_config['csv_file'], sep=" ")

        inputs_vectors = {}
        input_experiments = self.oghma_network_config['experimental']
        for experiment in input_experiments:
            points = len(self.oghma_network_config['experimental'][experiment]['vec']['points'].split(','))
            inputs_vectors[experiment] = points
            self.points = points
            self.population = len(training_dataset)

        self.inputs = self.oghma_network_config['sims'][self.networks_configured[self.working_network]]['inputs']
        self.outputs = self.oghma_network_config['sims'][self.networks_configured[self.working_network]]['outputs']

        self.input_points = 0
        for input in self.inputs:
            points = len(self.oghma_network_config['experimental'][input]['vec']['points'].split(','))
            self.input_points += points

        self.output_points = len(self.outputs)

        feature = np.zeros(self.input_points)
        features = np.zeros((self.population, self.input_points))
        target = np.zeros((len(self.outputs)))
        targets = np.zeros((self.population, self.output_points))

        inputs = np.empty(self.input_points, dtype=object)
        previous_end = 0
        for idx in range(len(self.inputs)):
            vector_points = inputs_vectors[self.inputs[idx]]
            inputs[previous_end:previous_end + vector_points] = np.asarray(
                [input + '.vec' + str(x) for x in range(vector_points)])
            previous_end = previous_end + vector_points

        self.features = training_dataset[inputs].to_numpy().astype(float)
        self.targets = training_dataset[self.outputs].to_numpy().astype(float)

        return features, targets

    def separate_training_dataset(self):
        features = np.asarray(self.features)
        targets = np.asarray(self.targets)
        training_population = int(self.model_settings.training_percentage * self.population)

        indices = np.linspace(0, self.population - 1, self.population, dtype=int)

        training_indices = random.choices(indices, k=training_population)
        validation_indices = np.delete(indices, training_indices)

        training_features = features[training_indices]
        training_targets = targets[training_indices]

        validation_features = features[validation_indices]
        validation_targets = targets[validation_indices]
        self.validation_indices = validation_indices
        self.training_features = training_features
        self.training_targets = training_targets
        self.validation_features = validation_features
        self.validation_targets = validation_targets

        return training_features, training_targets, validation_features, validation_targets

    def get_uniform_distribution(self, features, targets):
        features = np.asarray(features)
        targets = np.asarray(targets)

        training_population = int(self.model_settings.training_percentage * self.population)

        indices = np.linspace(0, self.population - 1, self.population, dtype=int)
        rng = np.random.default_rng()
        training_indices = rng.choice(indices, training_population, replace=False)
        validation_indices = np.array([i not in training_indices for i in indices])

        training_features = features[training_indices]
        training_targets = targets[training_indices]

        validation_features = features[validation_indices]
        validation_targets = targets[validation_indices]

        return validation_features, validation_targets

    def train_networks(self):
        dir = os.path.join(self.networks_dir, 'faster', self.networks_configured[self.working_network])
        self.networks[self.working_network] = Training(self.training_features, self.training_targets, self.validation_features, self.validation_targets, dir, model_settings=self.model_settings)

    def interpret_input_vectors(self):
        intensity = np.zeros(len(self.input_vectors))
        for idx, experiment in enumerate(self.input_vectors):
            match experiment:
                case x if 'light' in x:
                    experiment.split('_')
                    intensity[idx] = float(experiment.split('_')[-1])
                case x if 'dark' in x:
                    intensity[idx] = 0

    def sample_experimental_features(self, features):
        if features.Device_Population == 1:
            keys = self.oghma_network_config['sims'][self.networks_configured[self.working_network]]['inputs']
            keys = list(keys)
            l = 0
            for key in keys:
                l = l + len(self.input_vectors[key])
            filter = np.zeros(l)
            input = []
            for key in keys:
                input.append(self.input_vectors[key])
            input = np.array(input).ravel()
            for idx, i in enumerate(input):
                exp = features.x
                diff = exp - i
                diff = np.abs(diff)
                diff = np.argmin(diff)
                filter[idx] = np.abs(diff)
            filter = np.array(filter).astype(int)
            features.x = features.x[filter]
            features.y = features.y[filter]
        else:
            keys = self.oghma_network_config['sims'][self.networks_configured[self.working_network]]['inputs']
            keys = list(keys)
            l = 0
            for key in keys:
                l = l + len(self.input_vectors[key])
            filter = np.zeros(l)
            input = []
            for key in keys:
                input.append(self.input_vectors[key])
            input = np.array(input).ravel()
            for jdx in range(features.Device_Population):
                for idx, i in enumerate(input):
                    exp = features.x[jdx]
                    diff = exp - i
                    diff = np.abs(diff)
                    diff = np.argmin(diff)
                    filter[idx] = np.abs(diff)
                filter = np.array(filter).astype(int)
                features.x[jdx] = features.x[jdx][filter]
                features.y[jdx] = features.y[jdx][filter]

        return features

    def normalise_experimental_features(self, features, dir=None):
        if dir == None:
            min_max_log = pd.read_csv(os.path.join(self.networks_dir, 'faster', 'vectors', 'vec', 'min_max.csv'),
                                      header=None, sep=' ', names=['param', 'min', 'max', 'log'])
        else:
            min_max_log = pd.read_csv(os.path.join(dir, 'vectors', 'vec', 'min_max.csv'), header=None, sep=' ',
                                      names=['param', 'min', 'max', 'log'])
        vecs = ['light_1.0.vec' + str(i) for i in range(self.points)]
        min_max_log = min_max_log[min_max_log['param'].isin(vecs)]
        dir_min = min_max_log['min'].values
        dir_max = min_max_log['max'].values
        dir_log = min_max_log['log'].values
        y = features.y
        if len(y) != self.points:
            for idx in range(len(y)):
                if dir_log[0] == 0:
                    y[idx] = self.normalise_linear(y[idx], dir_min[idx], dir_max[idx])
                else:
                    y[idx] = self.normalise_log(y[idx], dir_min[idx], dir_max[idx])
        else:
            if dir_log[0] == 0:
                y = self.normalise_linear(y, dir_min, dir_max)
            else:
                y = self.normalise_log(y, dir_min, dir_max)

        features.y = y
        return features

    def denormalise_predictions(self):
        predictions = np.zeros((self.Device_Population, self.total_networks, 10))
        self.mean = np.zeros((self.total_networks, 10))
        for idx in range(self.total_networks):
            outputs = self.oghma_network_config['sims'][self.networks_configured[idx]]['outputs']
            num_outputs = len(outputs)
            network_min_max_log = self.min_max_log[self.min_max_log['param'].isin(outputs)]
            min = network_min_max_log['min'].values.ravel()
            max = network_min_max_log['max'].values.ravel()
            log = network_min_max_log['log'].values.ravel()

            for jdx in range(num_outputs):
                if num_outputs > 1:
                    if log[jdx] == 1:
                        predictions[:, idx, jdx] = self.denormalise_log(self.normalised_predicitons[idx][0][0][jdx], min[jdx], max[jdx])
                    else:
                        predictions[:, idx, jdx] = self.denormalise_linear(self.normalised_predicitons[idx][0][0][jdx], min[jdx], max[jdx])
                else:
                    if log == 1:
                        predictions[:, idx, jdx] = self.denormalise_log(self.normalised_predicitons[idx, jdx], min, max)
                    else:
                        predictions[:, idx, jdx] = self.denormalise_linear(self.normalised_predicitons[idx, jdx], min, max)
        self.predicitons = predictions

        for idx in range(self.total_networks):
            outputs = self.oghma_network_config['sims'][self.networks_configured[idx]]['outputs']
            num_outputs = len(outputs)
            for jdx in range(num_outputs):
                if num_outputs > 1:
                    self.mean[idx, jdx] = np.mean(self.predicitons[:, idx, jdx])
                else:
                    self.mean[idx,0] = np.mean(self.predicitons[:, idx, jdx])

    def normalise_linear(self, x, x_min, x_max):
        return (x - x_min) / (x_max - x_min)

    def normalise_log(self, x, x_min, x_max):
        return (np.log10(x) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))

    def denormalise_linear(self, x, x_min, x_max):
        return x * (x_max - x_min) + x_min

    def denormalise_log(self, x, x_min, x_max):
        return 10 ** (x * (np.log10(x_max) - np.log10(x_min)) + np.log10(x_min))


class Point(Networks):
    _network_type = 'point'

    def __init__(self, networks_dir, model_settings=None):
        self.networks_dir = networks_dir
        if model_settings == None:
            self.model_settings = Model_Settings()
        else:
            self.model_settings = model_settings
        self.rng = np.random.default_rng()

        self.min_max_log = pd.read_csv(os.path.join(self.networks_dir, 'faster', 'vectors', 'vec', 'min_max.csv'), header=None, sep=' ', names=['param', 'min', 'max', 'log'])

    def train(self):

        self.setup_network_directories()

        for idx in range(self.total_networks):
            self.working_network = idx

            self.load_training_dataset()
            self.separate_training_dataset()
            self.train_networks()

    def confusion_matrix(self):

        self.setup_network_directories()
        self.normalised_predicitons = np.zeros((len(self.networks_configured), 1), dtype=object)
        self.predictions = np.zeros((len(self.networks_configured), 10))
        self.MAPE = np.zeros((len(self.networks_configured),10))

        min_max_log = pd.read_csv(os.path.join(self.networks_dir, 'faster', 'vectors', 'vec', 'min_max.csv'),
                                  header=None, sep=' ', names=['param', 'min', 'max', 'log'])

        for idx in range(len(self.networks_configured)):
            self.working_network = idx

            self.load_training_dataset()
            self.separate_training_dataset()

            outputs = self.oghma_network_config['sims'][self.networks_configured[idx]]['outputs']
            network_min_max_log = min_max_log[min_max_log['param'].isin(outputs)]
            min = network_min_max_log['min'].values[0]
            max = network_min_max_log['max'].values[0]
            log = network_min_max_log['log'].values[0]

            figname = 'tempCF' + str(idx)
            fig, ax = plt.subplots(figsize=(6,6), dpi=300)
            ax.set_xlabel('Target')
            ax.set_ylabel('Predicted')
            dir = os.path.join(self.networks_dir, 'faster', self.networks_configured[self.working_network])
            validation_features = np.array([i.astype(float) for i in self.validation_features])
            P = Predicting(dir, validation_features)
            self.normalised_predicitons[idx,0] = P.predict()
            if np.shape(self.normalised_predicitons[idx][0])[1] > 1:
                for jdx in range(np.shape(self.normalised_predicitons[idx][0])[1]):
                    plt.hist2d(self.validation_targets[:,jdx].ravel(), self.normalised_predicitons[idx][0][:,jdx].ravel(), bins=np.linspace(0,1, 100), range=[[0, 1], [0, 1]], cmap='inferno')
                    self.MAPE[idx,jdx] = np.abs(np.mean(np.abs(self.validation_targets[:,jdx].ravel() - self.normalised_predicitons[idx][0][:,jdx].ravel()) /self.normalised_predicitons[idx][0][:,jdx].ravel()) * 100)
            else:
                plt.hist2d(self.validation_targets[:].ravel(), self.normalised_predicitons[idx][0].ravel(), bins=np.linspace(0, 1, 100), range=[[0, 1], [0, 1]], cmap='inferno')

                self.MAPE[idx,0] = np.abs(np.mean(np.abs(self.validation_targets.ravel() - self.normalised_predicitons[idx][0].ravel()) / self.normalised_predicitons[idx][0].ravel()) * 100)

            if not os.path.isdir(os.path.join(os.getcwd(), 'temp')):
                os.mkdir(os.path.join(os.getcwd(), 'temp'))
            figname = os.path.join(os.getcwd(), 'temp', figname)
            plt.savefig(figname)

        return self.MAPE

    def predict(self, experimental_feature):
        self.Device_Population = experimental_feature.Device_Population
        self.setup_network_directories()
        self.normalised_predicitons = np.zeros((len(self.networks_configured), 1), dtype=object)
        self.predictions = np.zeros((len(self.networks_configured), 10))
        for idx in range(len(self.networks_configured)):
            ef = copy.deepcopy(experimental_feature)
            self.working_network = idx

            dir = os.path.join(self.networks_dir, 'faster', self.networks_configured[self.working_network])

            self.load_input_vectors()
            self.interpret_input_vectors()

            ef = self.sample_experimental_features(ef)
            ef = self.normalise_experimental_features(ef)

            if ef.Device_Population > 1:
                ef.y = np.array([f for f in ef.y])
            else:
                ef.y = np.array([ef.y])

            P = Predicting(dir, ef.y)

            self.normalised_predicitons[idx,0] = P.predict()
        self.denormalise_predictions()
