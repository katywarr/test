import numpy as np
import pandas as pd
from utils.excel_handler import ExcelHandler, get_timestamp_str
from utils.topology import ESAMTopology
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import math
from matplotlib.ticker import MaxNLocator
from utils.probabilities import prob_f_h_carries_signal_fn, prob_post_conns_fn, prob_activation_fn, prob_firing_fn, \
    prob_correct_hoff
from utils.file_handler import create_dir


class TestSuiteResults:
    """Convenience class to plot a set of previously run tests"""

    def __init__(self,
                 test_dir: str,
                 test_suite_type: str,
                 test_name_mapping: dict,
                 test_data_sheet: str = 'latest_data',
                 plot_params_sheet: str = 'plot_params'
                 ):
        """
        Initialise the TestSuitePlotter with the relevant data.

        :param test_dir:
            A string describing the directory containing the Excel files that contain the data.
        :param test_suite_type:
            The test suite is either of type 'vary_network', in which case the tests have stable problem places and
            one or more network attributes is varied, or 'vary_problem', in which case the tests have a stable
            networks and one or more problem space attributes is varied.
        :param test_files:
            A list of strings indicating the tests in the test suite. Each test equates to the name of each file,
            without the extension. If this param is omitted, all the '.xlsx' files in the directory will be plotted.
            To ensure that plots occur in the correct order, specify this list explicitly.
        :param test_data_sheet:
            The name of the sheet in each file where the tests are located. This defaults to 'latest_data'. The name of
            the sheet must be the same across each of the files.
        :param plot_params_sheet:
            The name of the sheet in each file where the plot_params are located. This defaults to 'plot_params'.
            This sheet provides hints regarding how the data should be plotted.
        """
        # Check that the base folder exists
        if not os.path.exists(test_dir):
            print('Error: TestSuitePlotter - Directory containing the test data does not exist: {} '
                  '\nCurrent working directory is: {}'
                  .format(test_dir, os.getcwd()))
            return

        self.test_dir = test_dir        # Needed later for lazy getting of theoretical probabilities
        self.test_suite_type = test_suite_type
        if test_suite_type == 'vary_network':
            self.vary_col_prefix = 'net_param_'
            default_sheet_name = 'network_default'
        else:
            if test_suite_type == 'vary_problem':
                self.vary_col_prefix = 'data_param_'
                default_sheet_name = 'problem_default'
            else:
                raise ValueError('Error: test suite type of \'vary_network\' or \'vary_problem\' must be passed to the'
                                 ' test suite plotter.')
        self.test_suite_type = test_suite_type

        test_files = list(test_name_mapping.keys())
        self.num_tests = len(test_files)
        # Get a printable name for each of the tests
        if test_name_mapping is None:
            self.test_names = test_files
        else:
            self.test_names = []
            for test, test_name in test_name_mapping.items():
                self.test_names.append(test_name)
            if len(self.test_names) != len(test_files):
                msg = 'Error: The tests must correspond exactly to the dictionary keys in the test name mappings. \n ' \
                      'Tests are: {}\n' \
                      'Dictionary of test names is {}'.format(test_files, test_name_mapping)
                raise ValueError(msg)


        self.test_files = test_files
        # Check each of the test files contains data and populate lists with the required data.
        self.plot_params = []
        self.defaults_for_varying_params = []
        self.tests_data = []

        print('tests are : ', self.test_names)
        print('test dir: ', test_dir)
        for test in test_files:
            # The test_excel_data will handle any errors and throw exceptions. Let these trickle back to the caller.
            test_excel_data = ExcelHandler(file_dir=test_dir, file_name=test)
            self.plot_params.append(test_excel_data.read_sheet(sheet_name=plot_params_sheet).T.to_dict()[0])
            test_defaults = test_excel_data.read_sheet(sheet_name=default_sheet_name).T.to_dict()[0]
            if self.test_suite_type == 'vary_problem':
                # The number of epochs is not in the problem space defaults. Add this manually from the static network
                # data. Not all that elegant, but it works later.
                static_network = test_excel_data.read_sheet(sheet_name='network_params_static').T.to_dict()[0]
                test_defaults['e'] = static_network['e']

            self.defaults_for_varying_params.append(test_defaults)
            test_results = test_excel_data.read_sheet(sheet_name=test_data_sheet)
            # Remove epoch 0 - this is the test definition epoch which is not required for the runtime plots
            self.tests_data.append(test_results.copy().loc[test_results['epoch'] != 0])

        print('\nTestSuitePlotter successfully initialised for tests {}'.format(self.test_names))

    def get_theory_empirical_probs(self):

        theory_data = self.__get_theoretical_probabilities()

        tests_filtered = []
        for test_df, test_name, theory_df in zip(self.tests_data, self.test_names, theory_data):
            # Empirical data
            # It's just the first epoch we are interested in, filter on this one
            test_df_new = test_df.copy().loc[(test_df['epoch'] == 1)]
            test_df_new['Theory or empirical'] = np.repeat('empirical', test_df_new.shape[0])
            # Add new probability columns so that they all match perfectly (required for concat and plotting below)
            test_df_new['h_correct_prob'] = test_df_new['prop_h_gt']
            test_df_new['h_incorrect_prob'] = test_df_new['prop_h_not_gt']
            test_df_new['prob_correct'] = test_df_new['prop_correct']
            test_df_new['test_name'] = test_name

            # Theoretical data
            test_df_theory_new = theory_df.copy()
            test_df_theory_new['Theory or empirical'] = np.repeat('theory', test_df_theory_new.shape[0])

            # Combine and add to the list for plotting
            test_all = pd.concat([test_df_new, test_df_theory_new])
            tests_filtered.append(test_all)

        return tests_filtered

    def __get_theoretical_probabilities(self):

        df_theory_list = []     # A list of data frames, one per test, containing theoretical calculations

        # Pull out the relevant test description variations
        for plot_params, network_params, test_df, test_file_name in zip(self.plot_params,
                                                                   self.defaults_for_varying_params,
                                                                   self.tests_data,
                                                                   self.test_files):
            # The test_excel_data will handle any errors and throw exceptions. Let these trickle back to the caller.
            test_excel_data = ExcelHandler(file_dir=self.test_dir, file_name=test_file_name)
            if test_excel_data.check_sheet_name('theory'):
                print('Reading theoretical probabilities for test {} from file.'.format(test_file_name))
                df_theory = test_excel_data.read_sheet(sheet_name='theory')
            else:
                print('Calculating theoretical probabilities for test {}.'.format(test_file_name))

                # For this test, get the unique values from our variable column and put in a numpy array
                x_col = self.vary_col_prefix + plot_params['variable_column']
                variable_data = np.unique(test_df[x_col])
                # Create a data frame to represent each of the variations for this test. One row per variation.
                num_rows = len(variable_data)
                df_theory = pd.DataFrame()
                # Copy the data required for plotting/future calculations
                for row_num, variable in zip(range(num_rows), variable_data):
                    df_row = pd.DataFrame([test_df.loc[test_df[x_col] == variable].iloc[0]])
                    df_row_filtered = df_row.copy()
                    signal_fn = prob_f_h_carries_signal_fn(s_m=df_row_filtered['data_param_s_m'].iloc[0],
                                                           s_n=df_row_filtered['data_param_s_n'].iloc[0])
                    l_m = round(df_row_filtered['data_param_s_m'].iloc[0] * df_row_filtered['net_param_f'].iloc[0])
                    conn_fn = prob_post_conns_fn(length=l_m,
                                                 sparsity=df_row_filtered['net_param_f_h_sparsity'].iloc[0],
                                                 conn_type=df_row_filtered['net_param_f_h_conn_type'].iloc[0])
                    act_fn = prob_activation_fn(max_connections=l_m, signal_fn=signal_fn, connection_fn=conn_fn)
                    firing_fn = prob_firing_fn(max_conns=l_m, activation_fn=act_fn)
                    print('   Calculating theoretical probabilities for {} = {} of {}'
                          .format(x_col, variable, variable_data))

                    h_correct_prob = firing_fn(recall_type='correct',
                                               theta=df_row_filtered['net_param_h_thresh'].iloc[0])
                    h_incorrect_prob = firing_fn(recall_type='incorrect',
                                                 theta=df_row_filtered['net_param_h_thresh'].iloc[0])
                    df_row_filtered['h_correct_prob'] = h_correct_prob
                    df_row_filtered['h_incorrect_prob'] = h_incorrect_prob
                    inhib_value = df_row_filtered['net_param_h_f_sparsity_i'].iloc[0]
                    if inhib_value != 0:
                        inhib = True
                        if inhib_value != 1:
                            print('Warning: Setting inhibition {} to 1 for theoretical calculations.'
                                  .format(inhib_value))
                            df_row_filtered.loc[0, 'net_param_h_f_sparsity_i'] = 1
                            # df_row_filtered['net_param_h_f_sparsity_i'].iloc[0] = 1
                    else:
                        inhib = False
                    df_row_filtered['prob_correct'] = prob_correct_hoff(m=df_row_filtered['data_param_m'].iloc[0],
                                                                        h=df_row_filtered['net_param_h'].iloc[0],
                                                                        prob_h_firing_correct=h_correct_prob,
                                                                        prob_h_firing_incorrect=h_incorrect_prob,
                                                                        inhibition=inhib)
                    df_theory = pd.concat([df_theory, df_row_filtered])

                # Save the data for next time
                test_excel_data.add_sheet(df=df_theory, sheet_name='theory')

            # Add the data to the list of theory dataframes
            df_theory_list.append(df_theory)

        return df_theory_list
