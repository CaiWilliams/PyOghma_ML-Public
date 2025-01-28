import os
import re

import numpy as np
import pandas as pd
import shutil


from .Latex import Document
from .Figures import Figures
from .Networks import Networks
from .Labels import Label

class Output:
    def __init__(self, networks, inputs, abs_dir=None):
        pd.set_option('styler.format.precision', 3)
        self.networks = networks
        self.inputs = inputs
        self.figures = Figures()
        if abs_dir != None:
            self.abs_dir = abs_dir
        self.number_of_inputed_devices = 0
        self.number_of_networks_trained = len(self.networks.networks_configured)
        self.temp_dir = os.path.join(os.getcwd(), 'temp')
        if not os.path.isdir(self.temp_dir):
            os.mkdir(self.temp_dir)

    def build_report(self):
        self.pdf = Document(document_class='article', document_properties=['a4paper'], packages=['graphicx', 'geometry', 'booktabs', 'caption', 'subcaption', 'float', 'xcolor','colortbl', 'fancyhdr'])
        self.pdf.geometry(left='1cm', right='1cm', top='1.5cm', bottom='2cm')

        self.pdf.write(self.pdf.command('pagestyle', 'fancy'))
        self.pdf.write(self.pdf.command('lhead', 'Device Report'))
        self.pdf.write(self.pdf.command('rhead', '\\thepage'))
        self.pdf.write(self.pdf.command('cfoot','Report Produced by Oghma\_ML, Developed by Cai Williams'))

        self.pdf.begin_document()

        self.experimental_results()

        self.machine_learning_results()

    def experimental_results(self):
        self.pdf.section('Experimental Results')

        cap = 'Experimental Characteristics'
        characteristic = Figures()

        match self.inputs.characterisation_type:
            case 'JV'|'IV'|'JV_I4':
                characteristic.initialise_figure(figsize=(6, 6))

                characteristic.plot(self.inputs.x, self.inputs.y)

                characteristic.set_x_limits(left=np.min(self.inputs.x), right=np.max(self.inputs.x))
                characteristic.set_y_limits(top=np.max(self.inputs.y), bottom=np.min(self.inputs.y)*1.5)

                characteristic.set_x_label('Voltage (V)')
                characteristic.set_y_label('Current Density ($Am^{-2}$)')
            case 'batch_JV':
                characteristic.initialise_figure(figsize=(6, 6))

                for idx in self.inputs.x:
                    characteristic.plot(self.inputs.x, self.inputs.y)

                characteristic.set_x_limits(left=np.min(self.inputs.x), right=np.max(self.inputs.x))
                characteristic.set_y_limits(top=np.max(self.inputs.y), bottom=np.min(self.inputs.y) * 1.5)

                characteristic.set_x_label('Voltage (V)')
                characteristic.set_y_label('Current Density ($Am^{-2}$)')
            case '2d_JV':
                pass
            case _:
                pass

        characteristic_path = os.path.join(self.temp_dir, 'characteristic.png')
        characteristic.save_to_disk(characteristic_path, dpi=300)
        experimental_paramaters = self.calcluate_experimental_paramaters()

        cap = 'Experimental Input Characteristic'
        self.pdf.figure(characteristic_path, centering=True, width='0.6\\textwidth', caption=cap)

        cap = 'Parameters calculable from input characteristic'
        self.pdf.table(experimental_paramaters, centering=True, caption=cap)

        self.pdf.newpage()

    def calcluate_experimental_paramaters(self):
        match  self.inputs.characterisation_type:

            case 'JV' | 'JV_I4':
                V = self.inputs.x
                J = self.inputs.y

                Jsc = -J[np.argmin(np.abs(V-0))]
                Voc = V[np.argmin(np.abs(J-0))]

                P_max = np.abs(np.min(V*J))

                FF = P_max / (Voc * Jsc)

                PCE = Jsc * Voc * FF / 10

                experimental_parameters = {}

                experimental_parameters['Jsc'] = {}
                self.set_dictionary(experimental_parameters['Jsc'], Jsc, '$Am^{-2}$')

                experimental_parameters['Voc'] = {}
                self.set_dictionary(experimental_parameters['Voc'], Voc, '$V$')

                experimental_parameters['FF'] = {}
                self.set_dictionary(experimental_parameters['FF'], FF, '$a.u$')

                experimental_parameters['Pmax'] = {}
                self.set_dictionary(experimental_parameters['Pmax'], P_max, '$Wm^{-2}$')

                experimental_parameters['PCE'] = {}
                self.set_dictionary(experimental_parameters['PCE'], PCE, '$Percent$')

                experimental_parameters = pd.DataFrame(data=experimental_parameters).T
                experimental_parameters = experimental_parameters.reset_index(names='Parameter')
                #experimental_parameters.astype(float).round(3)
            case 'IV':
                V = self.inputs.x
                J = self.inputs.y

                Jsc = -J[np.argmin(np.abs(V - 0))]
                Voc = V[np.argmin(np.abs(J - 0))]

                P_max = np.abs(np.min(V * J))

                FF = P_max / (Voc * Jsc)

                PCE = Jsc * Voc * FF / 10

                experimental_parameters = {}

                experimental_parameters['Jsc'] = {}
                self.set_dictionary(experimental_parameters['Jsc'], Jsc, '$Am^{-2}$')

                experimental_parameters['Voc'] = {}
                self.set_dictionary(experimental_parameters['Voc'], Voc, '$V$')

                experimental_parameters['FF'] = {}
                self.set_dictionary(experimental_parameters['FF'], FF, '$a.u$')

                experimental_parameters['Pmax'] = {}
                self.set_dictionary(experimental_parameters['Pmax'], P_max, '$Wm^{-2}$')

                experimental_parameters['PCE'] = {}
                self.set_dictionary(experimental_parameters['PCE'], PCE, '$Percent$')

                experimental_parameters = pd.DataFrame(data=experimental_parameters).T
                experimental_parameters = experimental_parameters.reset_index(names='Parameter')
        return experimental_parameters

    def machine_learning_results(self):
        self.pdf.section('Machine Learning Parameters')

        match self.networks.__class__.__name__:
            case 'Point':
                self.confusion_matrices('Point')
                self.single_results()
            case _:
                raise ValueError('Network Type Not Recognised!')


    def confusion_matrices(self, network_type):
        A = Networks.initialise(self.networks.networks_dir, network_type=network_type)
        self.MAPE = A.confusion_matrix()

    def clean_parameter(self, parameter):
        parameter = re.sub(r"[-()\"#/@;:<>{}=~|.?,_]", " ", parameter)
        return parameter

    def multipoint_row(self, parameter, mean, std, mape, predictions):
        parameter = Label(parameter)
        self.prediction_dictionary[parameter.english] = {}
        dictionary = self.prediction_dictionary[parameter.english]
        if mean > 1e3 or mean < 1e-3:
            mean = '{:.2e}'.format(mean)
        dictionary['Mean'] = mean
        dictionary['Standard Deviation'] = std
        dictionary['Units'] = parameter.units
        dictionary['MAPE (\%)'] = mape
        return

    def multipoint_row_single(self, parameter, mean, mape):
        keys = list(self.prediction_dictionary.keys())
        parameter = Label(parameter)
        print(parameter.english)
        print(mean)
        if parameter.english in keys:
            number = len(np.where(parameter.english in keys))
            parameter.english = parameter.english + ' (' + str(number) + ')'
        self.prediction_dictionary[parameter.english] = {}
        dictionary = self.prediction_dictionary[parameter.english]
        #if mean > 1e3 or mean < 1e-3:
        #    mean = '{:.2e}'.format(mean)
        dictionary['Mean'] = mean
        dictionary['Units'] = parameter.units
        dictionary['MAPE (\%)'] = mape
        return

    def point_row(self, parameter, point):
        self.prediction_dictionary[parameter] = {}
        dictionary = self.prediction_dictionary[parameter]
        dictionary['Value'] = point
        dictionary['Units'] = 'to implement'
        return

    def prediction_table(self):
        networks = self.networks.networks_configured

        inputs = [self.networks.oghma_network_config['sims'][self.networks.networks_configured[working_network]]['inputs'] for working_network in range(len(networks))]
        outputs = [self.networks.oghma_network_config['sims'][self.networks.networks_configured[working_network]]['outputs'] for working_network in range(len(networks))]

        inputs = np.asarray(inputs)
        outputs = np.asarray(outputs, dtype=object)
        #outputs = outputs[outputs != 0]
        self.prediction_dictionary = {}
        self.prediction_all_dict = {}
        rows = np.zeros(len(networks) * len(outputs.ravel()),dtype=object)

        for idx in range(len(networks)):
            for jdx in range(len(outputs[idx])):
                parameter = outputs[idx][0]
                if len(self.networks.predicitons) > 0:
                    predictions_mean = self.networks.mean[idx]
                    predictions_std = self.networks.std[idx]
                    predictions_mape = self.MAPE[idx,jdx]
                    self.multipoint_row(parameter, predictions_mean, predictions_std, predictions_mape)
                else:
                    predictions = predictions[0]
                    self.point_row(parameter, predictions)
        predictions = pd.DataFrame(self.prediction_dictionary).T
        predictions = predictions.reset_index(names='Parameters')
        self.prediction_table = predictions
        #predictions.astype(float).round(3)
        return predictions

    def prediction_table_single(self):
        networks = self.networks.networks_configured

        inputs = [self.networks.oghma_network_config['sims'][self.networks.networks_configured[working_network]]['inputs'] for working_network in range(len(networks))]
        outputs = [self.networks.oghma_network_config['sims'][self.networks.networks_configured[working_network]]['outputs'] for working_network in range(len(networks))]

        inputs = np.asarray(inputs)
        outputs = np.asarray(outputs, dtype=object)
        #outputs = outputs[outputs != 0]
        self.prediction_dictionary = {}
        self.prediction_all_dict = {}
        rows = np.zeros(len(networks) * len(outputs.ravel()),dtype=object)

        for idx in range(len(networks)):
            for jdx in range(len(outputs[idx])):

                if len(outputs[idx]) == 1:
                    parameter = outputs[idx][0]
                    predictions_mean = self.networks.mean[idx][0]
                    predictions_mape = self.MAPE[idx][0]
                    self.multipoint_row_single(parameter, predictions_mean, predictions_mape)
                else:
                    print(idx, jdx)
                    print(outputs[idx][jdx])
                    print(outputs)
                    parameter = outputs[idx][jdx]
                    predictions_mean = self.networks.mean[idx][jdx]
                    predictions_mape = self.MAPE[idx][jdx]
                    self.multipoint_row_single(parameter, predictions_mean, predictions_mape)
        predictions = pd.DataFrame(self.prediction_dictionary).T
        predictions = predictions.reset_index(names='Parameters')
        self.prediction_table = predictions
        #predictions.astype(float).round(3)
        return predictions

    def distributions(self):
        figdir = os.path.join(os.getcwd(), 'temp')
        files = os.listdir(figdir)
        files = [os.path.join(figdir,x) for x in files if 'tempDF' in x]
        files = np.sort(files)
        return files


    def confustions(self):
        figdir = os.path.join(os.getcwd(), 'temp')
        files = os.listdir(figdir)
        files = [os.path.join(figdir,x) for x in files if 'tempCF' in x]
        files = np.sort(files)
        return files

    def insert_plots(self, df_files, cf_files):
        outputs = [self.networks.oghma_network_config['sims'][self.networks.networks_configured[working_network]]['outputs'] for working_network in range(len(self.networks.networks_configured))]
        if len(df_files) == len(cf_files):
            for idx in range(len(df_files)):
                try:
                    self.pdf.subsection(Label(outputs[idx][0]).english)
                except:
                    self.pdf.subsection('Error')
                figs = [df_files[idx], cf_files[idx]]
                self.pdf.subfigure(*figs, width=0.455)
        else:
            raise ValueError("The length of distributions does not match that of confusion matrices")

    def insert_plot(self, cf_files):
        outputs = [self.networks.oghma_network_config['sims'][self.networks.networks_configured[working_network]]['outputs'] for working_network in range(len(self.networks.networks_configured))]
        for idx in range(len(cf_files)):
            try:
                self.pdf.subsection(Label(outputs[idx][0]).english)
            except:
                self.pdf.subsection('Error')
            figs = [cf_files[idx]]
            self.pdf.subfigure(*figs, width=0.455)


    def single_results(self):
        predictions = self.prediction_table_single()
        cap = 'Machine learning predictions by the single method'
        self.pdf.table(predictions, centering=True, caption=cap)
        cfFiles = self.confustions()
        self.insert_plot(cfFiles)


    def set_dictionary(self, dict, value, unit):
        dict['Value'] = value
        dict['Unit'] = unit
        return dict

    def clean_up(self):
        files = os.listdir(os.getcwd())
        files = [f for f in files if 'tempfile' in f]
        files = [f for f in files if f.endswith('.pdf') == False]
        files = [os.path.join(os.getcwd(), f) for f in files if f.endswith('.csv') == False]

        for f in files:
            os.remove(f)

        os.rename('tempfile.pdf', str(self.name)+'.pdf')

        shutil.rmtree(os.path.join(os.getcwd(), 'temp'))


    def save_report(self, name):
        self.name = name
        self.pdf.end_document()
        self.pdf.save_tex()
        self.pdf.compile()
        self.pdf.compile()
        self.clean_up()
        self.prediction_table.to_csv(self.name + '.csv')

