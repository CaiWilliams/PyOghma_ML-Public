import h5py
import os
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

class Input:
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def experiment(cls, device_dir, characterisation_type=None, source_laboratory=None):
        if source_laboratory not in cls.subclasses:
            raise ValueError('Source laboratory: {} Not recognized'.format(source_laboratory))
        return cls.subclasses[source_laboratory](device_dir, characterisation_type)

    def calcluate_JV_Params(self, voltage, current):
        V = voltage
        J = current

        self.Jsc = -J[np.argmin(np.abs(V - 0))]
        self.Voc = V[np.argmin(np.abs(J - 0))]

        self.P_max = np.abs(np.min(V * J))

        self.FF = self.P_max / (self.Voc * self.Jsc)

        self.PCE = self.Jsc * self.Voc * self.FF / 10
        return


class Deibel(Input):
    _source_laboratory = 'Deibel'
    def __init__(self, device_dir, characterisation_type):
        self.device_dir = device_dir
        self.characterisation_type = characterisation_type
        self.parse()

    def parse(self):
        match self.characterisation_type:
            case 'IV':
                self.current_voltage_area()
            case 'JV':
                self.current_voltage()
            case 'JV_I4':
                self.current_voltage_I4()
            case 'batch_JV':
                self.batch_current_voltage()
            case _:
                raise ValueError('The {} Laboratory does not support {}'.format(self._source_laboratory, self.characterisation_type))

    def batch_current_voltage(self):

        Device_Population = len(self.device_dir)
        self.Device_Population = Device_Population
        if Device_Population <= 1:
            raise ValueError('One or less devices detected!')
        else:
            voltage = np.zeros(Device_Population, dtype=object)
            current_density = np.zeros(Device_Population, dtype=object)
            metadata = np.zeros(Device_Population, dtype=object)

            for index in range(Device_Population):
                self.current_voltage(index)
                self.Device_Population = Device_Population
                voltage[index] = self.voltage
                current_density[index] = self.current_density
                metadata[index] = self.metadata

        self.voltage = voltage
        self.current_density = current_density
        self.metadata = metadata

    def current_voltage_area(self, index=None):
        self.Device_Population = 1
        metadata_list = []

        if index is not None:
            d = self.device_dir[index]
        else:
            d = self.device_dir

        with open(d, 'r') as f:
            for line in f:

                if line[0:9] == '##columns':
                    line = line[2:]
                    line = line.replace('\n', '')
                    line = line.replace(';', '')
                    #line = line.replace('=', ' ')
                    line = line.replace(' ',',')
                    metadata_list.append([line])
                else:
                    if line[0] == '#':
                        line = line[2:]
                        line = line.replace('\n', '')
                        line = line.replace('\t', '')
                        line = line.replace(' ', '')
                        metadata_list.append(line.split(';')[:-1])
        metadata_list = np.concatenate(metadata_list)

        group_metadata = {}
        for parameter in metadata_list:

            parameter = parameter.split('=')

            match parameter[0]:

                case 'SMU|ILLUMINATOR':
                    temporary_keys = parameter[0].split('|')
                    temporary_values = parameter[1].split('|')

                    for i in range(len(temporary_keys)):
                        group_metadata[temporary_keys[i]] = temporary_values[i]

                case 'columns':
                    parameter[-1] = parameter[-1].split(',')
                    group_metadata['columns'] = parameter[1:]
                    group_metadata['columns'] = group_metadata['columns'][0]

                case _:
                    temporary_keys = parameter[1].split(',')

                    if "/" in temporary_keys[0]:
                        temporary_keys = temporary_keys[0].split('/')

                    if len(temporary_keys) == 1:
                        group_metadata[parameter[0]] = temporary_keys[0]
                    else:
                        group_metadata[parameter[0]] = temporary_keys
        data = pd.read_csv(d, sep='\t', header=None, comment='#', names=group_metadata['columns'])

        voltage = data['V'].to_numpy()
        current_density = data['I'].to_numpy()
        current_density = current_density * 1000 / 0.04 * 10

        metadata = {}
        # light_source = group_metadata['ILLUMINATOR']
        # if group_metadata['incidentIntensity[%]'] == 'dark':
        #     group_metadata['incidentIntensity[%]'] = 0
        # elif group_metadata['incidentIntensity[%]'] == light_source:
        #     group_metadata['incidentIntensity[%]'] = 1
        #
        # match group_metadata['incidentIntensity[%]']:
        #     case 0:
        #         metadata['intensity'] = 0
        #     case 1:
        #         metadata['intensity'] = 1
        #     case _:
        #         metadata['intensity'] = group_metadata['incidentIntensity[%]'] / 100
        #
        # metadata = metadata

        self.voltage = voltage
        self.current_density = current_density
        self.metadata = metadata

    def current_voltage(self, index=None):
        self.Device_Population = 1
        metadata_list = []

        if index is not None:
            d = self.device_dir[index]
        else:
            d = self.device_dir

        with open(d, 'r') as f:
            for line in f:

                if line[0:9] == '##columns':
                    line = line[2:]
                    line = line.replace('\n', '')
                    line = line.replace(';', '')
                    #line = line.replace('=', ' ')
                    line = line.replace(' ',',')
                    metadata_list.append([line])
                else:
                    if line[0] == '#':
                        line = line[2:]
                        line = line.replace('\n', '')
                        line = line.replace('\t', '')
                        line = line.replace(' ', '')
                        metadata_list.append(line.split(';')[:-1])
        metadata_list = np.concatenate(metadata_list)

        group_metadata = {}
        for parameter in metadata_list:

            parameter = parameter.split('=')

            match parameter[0]:

                case 'SMU|ILLUMINATOR':
                    temporary_keys = parameter[0].split('|')
                    temporary_values = parameter[1].split('|')

                    for i in range(len(temporary_keys)):
                        group_metadata[temporary_keys[i]] = temporary_values[i]

                case 'columns':
                    parameter[-1] = parameter[-1].split(',')
                    group_metadata['columns'] = parameter[1:]
                    group_metadata['columns'] = group_metadata['columns'][0]

                case _:
                    temporary_keys = parameter[1].split(',')

                    if "/" in temporary_keys[0]:
                        temporary_keys = temporary_keys[0].split('/')

                    if len(temporary_keys) == 1:
                        group_metadata[parameter[0]] = temporary_keys[0]
                    else:
                        group_metadata[parameter[0]] = temporary_keys
        data = pd.read_csv(d, sep='\t', header=None, comment='#', names=group_metadata['columns'])

        voltage = data['V'].to_numpy()
        current_density = data['J'].to_numpy()

        metadata = {}
        light_source = group_metadata['ILLUMINATOR']
        if group_metadata['incidentIntensity[%]'] == 'dark':
            group_metadata['incidentIntensity[%]'] = 0
        elif group_metadata['incidentIntensity[%]'] == light_source:
            group_metadata['incidentIntensity[%]'] = 1

        match group_metadata['incidentIntensity[%]']:
            case 0:
                metadata['intensity'] = 0
            case 1:
                metadata['intensity'] = 1
            case _:
                metadata['intensity'] = group_metadata['incidentIntensity[%]'] / 100

        metadata = metadata

        self.voltage = voltage
        self.current_density = current_density
        self.metadata = metadata

    def current_voltage_I4(self, index=None):
        self.Device_Population = 1
        metadata_list = []

        if index is not None:
            d = self.device_dir[index]
        else:
            d = self.device_dir

        with open(d, 'r') as f:
            for line in f:

                if line[0:9] == '##columns':
                    line = line[2:]
                    line = line.replace('\n', '')
                    line = line.replace(';', '')
                    # line = line.replace('=', ' ')
                    line = line.replace(' ', ',')
                    metadata_list.append([line])
                else:
                    if line[0] == '#':
                        line = line[2:]
                        line = line.replace('\n', '')
                        line = line.replace('\t', '')
                        line = line.replace(' ', '')
                        metadata_list.append(line.split(';')[:-1])
        metadata_list = np.concatenate(metadata_list)

        group_metadata = {}
        for parameter in metadata_list:

            parameter = parameter.split('=')

            match parameter[0]:

                case 'SMU|ILLUMINATOR':
                    temporary_keys = parameter[0].split('|')
                    temporary_values = parameter[1].split('|')

                    for i in range(len(temporary_keys)):
                        group_metadata[temporary_keys[i]] = temporary_values[i]

                case 'columns':
                    parameter[-1] = parameter[-1].split(',')
                    group_metadata['columns'] = parameter[1:]
                    group_metadata['columns'] = group_metadata['columns'][0]

                case _:
                    temporary_keys = parameter[1].split(',')

                    if "/" in temporary_keys[0]:
                        temporary_keys = temporary_keys[0].split('/')

                    if len(temporary_keys) == 1:
                        group_metadata[parameter[0]] = temporary_keys[0]
                    else:
                        group_metadata[parameter[0]] = temporary_keys
        data = pd.read_csv(d, sep='\t', header=None, comment='#', names=group_metadata['columns'])

        voltage = data['V'].to_numpy()
        current = data['I'].to_numpy()

        voltage, current_density, light_intensity = self.I4_scaler(voltage, current)

        metadata = {}
        metadata['intensity'] = light_intensity


        self.voltage = voltage
        self.current_density = current_density
        self.metadata = metadata

    def I4_scaler(self, voltage, current):
        dir = os.path.dirname(self.device_dir)
        name = os.path.basename(self.device_dir)
        name = name.split('_')
        name = name[0:-1]

        illumination = name[-1].replace('uIllu', '')
        illumination = float(illumination) / 1e6

        name[-1] = 'sunsVoc.dat'
        name_sunsVoc = '_'.join(name)


        name[-1] = 'am15.dat'
        name = [name[0], name[-1]]

        name_JV = '_'.join(name)
        data_JV = pd.read_csv(os.path.join(dir,name_JV),comment='#', sep='\t', names = ['V', 'I', 'J'])
        JV_voltage = data_JV['V'].to_numpy()
        JV_current = data_JV['J'].to_numpy()
        self.calcluate_JV_Params(-JV_voltage, JV_current)
        self.Jsc = -self.Jsc

        data_sunsVoc = pd.read_csv(os.path.join(dir,name_sunsVoc),comment='#', sep='\t', names = ['filters','power','relIllu', 'Voc', 'Isc', 'Iphoto'])
        data_sunsVoc = data_sunsVoc.drop(columns=['power'])
        data_sunsVoc['Isc'] = data_sunsVoc['Isc'] * -1
        data_sunsVoc['Isc'] = np.log(data_sunsVoc['Isc'])
        function = interpolate.PchipInterpolator(data_sunsVoc['Voc'], data_sunsVoc['Isc'], extrapolate=True)
        y = np.linspace(0,0.8,1000)
        x = function(self.Voc)
        sfactor = (1e3/(self.Jsc))*(np.exp(x))
        current_density = current * (1e3/sfactor)

        data_sunsVoc['Iphoto'] = np.log(data_sunsVoc['Iphoto'])
        function = interpolate.PchipInterpolator(data_sunsVoc['Voc'], data_sunsVoc['Iphoto'], extrapolate=True)
        light_intensity = np.exp(data_sunsVoc['Iphoto']) / np.exp(function(self.Voc))

        function = interpolate.PchipInterpolator(data_sunsVoc['relIllu'], light_intensity, extrapolate=True)
        light_intensity = function(illumination)

        return voltage, current_density, light_intensity






    def standardise_inputs(self):
        self.x = np.array(self.voltage)
        self.y = np.array(self.current_density)
        self.metadata = self.metadata


class Herzig(Input):
    _source_laboratory = 'Herzig'

    def __init__(self, device_dir, characterisation_type):
        self.device_dir = device_dir
        self.characterisation_type = characterisation_type
        self.parse()

    def parse(self):
        match self.characterisation_type:
            case '2D_JV':
                self.mapped_current_voltage()
            case _:
                raise ValueError('The {} Laboratory does not support {}'.format(self._source_laboratory, self.characterisation_type))

    def current_voltage(self, index=None):

        if index is not None:
            d = self.data[index]
        else:
            raise ValueError('This function does not currently handel single JVs')

        metadata = {}
        metadata['intensity'] = self.intensity[index]

        self.voltage = -self.data[index,:,0]
        self.current = -self.data[index,:,1]
        self.metadata = metadata


    def mapped_current_voltage(self):
        data = h5py.File(self.device_dir, 'r')
        data = data['data']

        shape = data[:, :, 0, :, :].shape

        dark = data[:,:,0,:,:].reshape((shape[0]*shape[1],shape[2],shape[3]))
        dark_intensity = np.zeros(len(dark))

        light = data[:,:,2,:,:].reshape((shape[0]*shape[1],shape[2],shape[3]))
        light_intensity = np.ones(len(light))

        data = np.concatenate([dark, light])

        intensity = np.concatenate((dark_intensity, light_intensity))

        self.data = data
        self.intensity = intensity

        voltage = np.zeros((shape[0]*shape[1]*2), dtype=object)
        current = np.zeros((shape[0]*shape[1]*2), dtype=object)
        metadata = np.zeros((shape[0]*shape[1]*2), dtype=object)

        for index in range(int(shape[0]*shape[1]*2)):
            self.current_voltage(index)
            voltage[index] = self.voltage
            current[index] = self.current
            metadata[index] = self.metadata

        self.voltage = voltage
        self.current = current
        self.metadata = metadata


    def current_to_current_density(self, area=8e-7):
        self.current_density = self.current/area
        self.current_density_flat = True

    def standardise_inputs(self):
        self.x = self.voltage

        if self.current_density_flat:
            self.y = self.current_density
        else:
            self.y = self.current

        self.metadata = self.metadata


class Shoaee(Input):
    _source_laboratory = 'Shoaee'
    def __init__(self, device_dir, characterisation_type):
        raise ValueError('The {} Laboratory is currently under construction!'.format(self._source_laboratory))


class Oghma(Input):
    _source_laboratory = 'Oghma'

    def __init__(self, device_dir, characterisation_type):
        self.device_dir = device_dir
        self.characterisation_type = characterisation_type
        self.parse()


    def parse(self):
        match self.characterisation_type:
            case 'JV':
                self.current_voltage()
            case 'batch_JV':
                self.batch_current_voltage()
            case _:
                raise ValueError('The {} Laboratory does not support {}'.format(self._source_laboratory, self.characterisation_type))

    def current_voltage(self, index=None):
        self.Device_Population = 1
        metadata_list = []

        if index is not None:
            d = self.device_dir[index]
        else:
            d = self.device_dir

        data = pd.read_csv(d, sep=' ', header=None, comment='#', skiprows=2, names=['V', 'J'])

        voltage = data['V'].to_numpy()
        current_density = data['J'].to_numpy()

        metadata = {}


        self.voltage = voltage
        self.current_density = current_density
        self.metadata = metadata

    def standardise_inputs(self):
        self.x = np.array(self.voltage)
        self.y = np.array(self.current_density)
        self.metadata = self.metadata