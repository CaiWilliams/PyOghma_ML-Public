
# PyOghma_ML

A machine learning pipeline for OghmaNano.


## Installation

Prior to installing PyOghma it is critical that OghmaNano is installed. OghmaNano can be found to download at: [oghma-nano.com](https://www.oghma-nano.com)

PyOghma_ML may be installed using pip


```bash
    python -m pip install PyOghma-ML
```
    
## Usage/Examples

This guide is not a guide for how to utilise the machine learning features of OghmaNano. If you are unfamiliar with OghmaNano or its machine learning features please refer to the [OghmaNano User Manual](https://www.oghma-nano.com/docs/man/understanding_oghma_nano.pdf) before proceeding.

### Machine Learning Inputs

The machine learning networks to be trained or later used for predicticting from experimental characteristics are set up and defined through the machine learning window within OghmaNano.

With the dataset generated and vectors built and normalised, all files required to use the machine learning pipeline have been generated.

For refrence the files used within the pipeline are:
  - nets.json: Defines the networks to be trained, their inputs and outputs
  - vectors.csv: Defines the normalised value of each parameter for each device
  - min_max.csv: Defines the minimum value, maximum value, and the space used by each parameter

Of these files only nets.json may be modified after dataset and vectors generation however care must be taken if modified outside of OghmaNano.

### Training

To quickly train a prepared OghmaNano directory the below code may be referred to as a guide.

```python
import PyOghma_ML as OML
import os

dir = os.path.join('Path','to','OghmaNano','Directory)
A = OML.Networks.initialise(dir, network_type='Point')
A.train()
```
So far within this release version only one network_type is included 'Point' as it yields a point prediction from a deep neural network.

For finer control of the networks to be trained, the settings for the models used may be accessed, modified and passed to the initialiser. As can be seen below.

```python
import PyOghma_ML as OML
from PyOghma_ML.Training import Model_Settings
import os

dir = os.path.join('/','media','cai','Big','PycharmProjects','Simulations', 'ML_testing','Single')

M = Model_Settings()
M.layer_nodes = [16,16,16,16]
M.dropout = [0,0,0,0]

A = OML.Networks.initialise(dir, network_type='Point', model_settings=M)
A.train()
```

The default settings for model settings can be seen in the below table.

| Model Setting Setting | Default Value | Modification Guide |
|:-------:|:-------------:|:------------------|
|initializer | 'glorot_uniform' | [For valid kernel initailizers](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)|
|activation | 'gelu' | [For valid activation functions](https://www.tensorflow.org/api_docs/python/tf/keras/activations) |
|layer_nodes| [2048, 2048, 2048, 2048]| Number of nodes in each hidden layer |
|dropout| [0.01, 0.01, 0.01, 0.01]| Dropout fraction of each hidden layer |
|batch_norm| True | Boolean of whether batch normalisation is applied|
|epochs| 256 | Maximum number of training epochs|
|learning_rate| Decaying learning Rate Function | Static learning rates or [custom schedulers maybe applied](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/LearningRateSchedule) may be applied |
|batch_size| 2024 | Number of devices considered per batch of training |
|patience | 16 | Number of epochs with no improvment before training exits early |
|loss_function | 'mse' | [For valid loss functions](https://www.tensorflow.org/api_docs/python/tf/keras/losses)|
|metrics | [tf.keras.metrics.MeanAbsoluteError()]| List of tensorflow metrics functions, [For futher metrics functions](https://www.tensorflow.org/api_docs/python/tf/keras/metrics)|
|training_percentage| 0.8 | |
|validation_percentage | 0.2 | training_percentage + validation_percentage must equals 1|


### Experimental Inputs

In order to ensure a common format for the machine learning when handling data from various laboratories data input fuctions have been writen. For new laboratories or characteristics it may be required to write a data input function.

The characterisation type and source laboratory can be specified when initialising an input. 

```python
exp_dir = os.path.join('Path','to','Experimental','Data)

Exp = OML.Input.experiment(device_dir=exp_dir, characterisation_type='JV', source_laboratory='Deibel')
Exp.standardise_inputs()
```

The current valid source laboratories can be seen in the table below. To be added please contact.

|Laboratory|Label|
|----------|-----|
|OPKM | Deibel |
|HSP | Shoaee |
|Herzig Group | Herzig|
|OghmaNano| Oghma |


### Predictions

With a defined experimental input machine learning predictions may be made based upon this data. The below code may serve as a guide for generating predictions.

```python
import PyOghma_ML as OML
import os

ml_dir = os.path.join('Path','to','OghmaNano','Directory)
exp_dir = os.path.join('Path','to','Experimental','Data)

Exp = OML.Input.experiment(device_dir=exp_dir, characterisation_type='JV', source_laboratory='Deibel')
Exp.standardise_inputs()

A = OML.Networks.initialise(ml_dir, network_type='Point')

A.predict(Exp)

O = OML.Output(A, Exp)
O.build_report()
O.save_report('DeviceReport')
```

By default, PyOghma_ML will output both a latex based report of predictions accompanied by the networks respective confusion matrices and csv table of the predicted values.
On Linux the latex report will automatically compile utilising pdflatex and then component files deleted. On Windows this is a manual process.

## Authors

- [@CaiWilliams](https://github.com/CaiWilliams)

