import PyOghma-ML as OML
import os

ml_dir = os.path.join('/', 'media', 'cai', 'Big', 'PycharmProjects', 'Simulations', 'ML_testing','Single')
exp_dir = os.path.join('/', 'media', 'cai', 'Big', 'Experimental Data', 'Chen', 'IV', 'Look At', '04-08-2024 DA ratio', '45pPM6s1_am15.dat')

Exp = OML.Input.experiment(device_dir=exp_dir, characterisation_type='JV', source_laboratory='Deibel')
Exp.standardise_inputs()

A = OML.Networks.initialise(ml_dir, network_type='Point')

A.predict(Exp)

O = OML.Output(A, Exp)
O.build_report()
O.save_report('DeviceReport')