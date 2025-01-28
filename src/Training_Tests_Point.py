import PyOghma_ML as OML
from PyOghma_ML.Training import Model_Settings
import os

dir = os.path.join('/','media','cai','Big','PycharmProjects','Simulations', 'ML_testing','Single')

M = Model_Settings()
M.layer_nodes = [16,16,16,16]
M.dropout = [0,0,0,0]

A = OML.Networks.initialise(dir, network_type='Point', model_settings=M)
A.train()
