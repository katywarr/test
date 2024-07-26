import numpy as np
import pandas as pd
from abc import ABCMeta
from pathlib import Path
import os
from utils.excel_handler import ExcelHandler
from esam.connections import ConnectionType, get_connection_type
from utils.probabilities import n_choose_k_with_prob, prob_f_h_carries_signal, n_choose_fewer_than_k_with_prob

# These column names are used in for the cached hyperparameter data - the minimum data that must be saved
# to generate the accuracy probabilities and to generate the best threshold.
# Therefore, the names must not change. Any change will invalidate any previously cached data.
pd_col_name_f = 'net_param_f'
pd_col_name_h = 'net_param_h'
pd_col_name_f_h = 'net_param_f_h_sparsity'
pd_col_name_f_h_conn_type = 'net_param_f_h_conn_type'
pd_col_name_s = 'data_param_s_m'
pd_col_name_n = 'data_param_s_n'
pd_col_name_t = 'threshold'
pd_col_name_h_perfect_prob_act = 'h_perfect_prob_act'
pd_col_name_h_correct_prob_act = 'h_correct_prob_act'
pd_col_name_h_incorrect_prob_act = 'h_incorrect_prob_act'
pd_col_name_h_perfect_prob_t = 'h_perfect_prob_t'
pd_col_name_h_correct_prob_t = 'h_correct_prob_t'
pd_col_name_h_incorrect_prob_t = 'h_incorrect_prob_t'
# Cache file name and directory also remains constant
cache_file_dir = '..' + os.sep + 'hyperparameter_data'
cache_file_name = 'hidden_neuron_activations'
cache_file_sheet = 'Sheet1'


class HiddenNeuronParameters(metaclass=ABCMeta):

    def __init__(self, problem_space: dict,
                 net_param_h: int,
                 net_param_f_h_conn_type: str,
                 net_param_f_h_sparsity: float,
                 use_cached_data: bool = True,
                 debug: bool = False):
        """
        :param network_params:
        f_h_connection_prob - the probability of a learned connection being established between a feature neuron
        and a hidden neuron. As the connections always represent a feature neuron containing a signal (1), this
        is adjusted in the code to be f_h_connection_prob/sparsity.
        signal_sparsity - probability between 0 an 1 indicating the proportion of 1s in the signal.
        """
        self.debug = debug
        self.net_param_f = problem_space['f']
        self.net_param_h = net_param_h          # Just for recording
        if net_param_f_h_sparsity == 0:
            raise ValueError('Unable to generate threshold data. Feature to hidden neuron probability cannot be zero.')
        self.net_param_f_h_sparsity = net_param_f_h_sparsity
        self.data_param_s_m = problem_space['s_m']
        self.data_param_s_n = problem_space['s_n']
        self.net_param_f_h_conn_type = net_param_f_h_conn_type

        self.f_h_learned_connection_prob = self.net_param_f_h_sparsity * self.data_param_s_m
        # -------------------------------------------------------------------------------------------------------------
        # Probabilities that a learned feature-hidden unit connection will be activated
        # -------------------------------------------------------------------------------------------------------------
        # With a perfect memory, the probability that an f-h learned connection is activated is 1 because all the
        # learned connections will be activated by the perfect signal.
        signal_prob_perfect_memory = 1
        signal_prob_noisy_memory = prob_f_h_carries_signal('correct', s_m=self.data_param_s_m, s_n=self.data_param_s_n)
        signal_prob_noisy_unknown = prob_f_h_carries_signal('incorrect', s_m=self.data_param_s_m, s_n=self.data_param_s_n)

        if self.debug:
            print('   signal_prob_perfect_memory = ', signal_prob_perfect_memory)
            print('   signal_prob_noisy_memory   = ', signal_prob_noisy_memory)
            print('   signal_prob_noisy_unknown  = ', signal_prob_noisy_unknown)

        # -------------------------------------------------------------------------------------------------------------
        # Pandas Datatype Column Names
        # -------------------------------------------------------------------------------------------------------------
        self.pd_col_name_activation_level = 'Activation or Threshold'
        self.pd_col_name_activation_prob = 'Activation Probability'
        self.pd_col_name_threshold_prob = 'Threshold Probability'
        self.pd_col_name_data_type = 'Data Presented to Hidden Neuron'
        # Columns that depend on the number of memories
        self.pd_col_name_activation_prob_scaled = 'Activation Probability Scaled'
        self.pd_col_name_threshold_prob_scaled = 'Threshold Probability Scaled'

        if use_cached_data:
            if self.__check_for_data_in_cache():
                return          # Cached data successfully de-serialised

        # -------------------------------------------------------------------------------------------------------------
        # Calculate the threshold and activation probabilities for all hidden neuron cases
        # This generates 3 pandas data frames
        # -------------------------------------------------------------------------------------------------------------
        print('Generating probability data for test ...')
        # A hidden neuron gets a pattern that it has previously learned.
        if self.debug:
            print('  Calculating activation probabilities for perfect memory recall')
        self.probs_perfect_recall = self.__calc_hidden_neuron_activation_probabilities(
            probability=signal_prob_perfect_memory,
            data_type='Perfect memory recall')

        # A hidden neuron gets a pattern that it has learned, but the pattern has noise.
        if self.debug:
            print('  Calculating activation probabilities for correct recall')
        self.probs_correct_recall = self.__calc_hidden_neuron_activation_probabilities(
            probability=signal_prob_noisy_memory,
            data_type='Correct recall')

        # A hidden neuron gets an unknown pattern.
        if self.debug:
            print('  Calculating activation probabilities for incorrect recall')
        self.probs_incorrect_recall = self.__calc_hidden_neuron_activation_probabilities(
            probability=signal_prob_noisy_unknown,
            data_type='Incorrect recall')

        cache_format_data = self.get_probability_data_for_cache()
        print('   ...Probability data generated')

        if use_cached_data:
            print('    Saving the data calculated for future in the cache file.')
            writer = ExcelHandler(cache_file_dir, cache_file_name)
            writer.add_rows(cache_format_data, cache_file_sheet)

    def __check_for_data_in_cache(self):

        full_file_name = cache_file_dir + os.sep + cache_file_name + '.xlsx'
        print('    Checking for previously cached probability results in file {}'.format(full_file_name))

        if Path(full_file_name).is_file():
            cached_all = pd.read_excel(full_file_name, sheet_name=cache_file_sheet)
            cached_probabilities = \
                cached_all[(cached_all['net_param_f'] == self.net_param_f) &
                           (cached_all['net_param_h'] == self.net_param_h) &
                           (cached_all['net_param_f_h_sparsity'] == self.net_param_f_h_sparsity) &
                           (cached_all['net_param_f_h_conn_type'] == self.net_param_f_h_conn_type) &
                           (round(cached_all['data_param_s_m'], 4) == round(self.data_param_s_m, 4)) &
                           (cached_all['data_param_s_n'] == self.data_param_s_n)]
            if cached_probabilities.shape[0] > 0:
                print('    Using cached probability data from {}'.format(full_file_name))
            else:
                print('    No matching probability data found in the cache file {}'.format(full_file_name))
                return False
        else:
            print(
                '    WARNING: Cache probability file {} not found. Threshold will be generated '
                'and saved if the cache dir exists.'
                '\n      If the cache file exists, and this warning is unexpected,  '
                'check the working directory. Current working directory is: {}'
                .format(full_file_name, os.getcwd()))
            return False

        # De-serialise data from the cache
        self.probs_perfect_recall = self.generate_h_probs_df(
            activations=cached_probabilities['threshold'].to_numpy(),
            activation_probabilities=cached_probabilities[pd_col_name_h_perfect_prob_act].to_numpy(),
            threshold_probabilities=cached_probabilities[pd_col_name_h_perfect_prob_t].to_numpy(),
            recall_type='Perfect memory recall')
        self.probs_correct_recall = self.generate_h_probs_df(
            activations=cached_probabilities['threshold'].to_numpy(),
            activation_probabilities=cached_probabilities[pd_col_name_h_correct_prob_act].to_numpy(),
            threshold_probabilities=cached_probabilities[pd_col_name_h_correct_prob_t].to_numpy(),
            recall_type='Correct recall')
        self.probs_incorrect_recall = self.generate_h_probs_df(
            activations=cached_probabilities['threshold'].to_numpy(),
            activation_probabilities=cached_probabilities[pd_col_name_h_incorrect_prob_act].to_numpy(),
            threshold_probabilities=cached_probabilities[pd_col_name_h_incorrect_prob_t].to_numpy(),
            recall_type='Incorrect recall')

        return True

    def generate_h_probs_df(self,
                            activations: np.ndarray,
                            activation_probabilities: np.ndarray,
                            threshold_probabilities: np.ndarray,
                            recall_type: str):
        h_probs_df = pd.DataFrame(activations, columns=[self.pd_col_name_activation_level])
        h_probs_df[self.pd_col_name_activation_prob] = activation_probabilities
        h_probs_df[self.pd_col_name_threshold_prob] = threshold_probabilities
        h_probs_df[self.pd_col_name_data_type] = np.full(shape=activations.shape[0],
                                                                fill_value=recall_type)
        h_probs_df.fillna(0, inplace=True)
        return h_probs_df

    def get_probabilities_for_threshold(self, threshold) -> (float, float, float):
        return self.probs_perfect_recall[self.pd_col_name_threshold_prob][threshold], \
            self.probs_correct_recall[self.pd_col_name_threshold_prob][threshold], \
            self.probs_incorrect_recall[self.pd_col_name_threshold_prob][threshold]

    def get_hidden_neuron_spike_confidence_perfect(self, threshold):
        """
        Return the confidence that a hidden neuron will spike when the system is presented with a perfect memory
        that it has learned given a specific threshold. This assumes a single hidden neuron.
        :param threshold:
        Hidden neuron threshold.
        :return:
        Confidence value
        """
        return self.probs_perfect_recall[self.pd_col_name_threshold_prob][threshold]

    def get_hidden_neuron_spike_confidence_noisy(self, threshold):
        """
        Return the confidence that a hidden neuron will spike when the system is presented with a noisy memory
        that it has learned given a specific threshold. This assumes a single hidden neuron.
        :param threshold:
        Hidden neuron threshold.
        :return:
        Confidence value
        """
        return self.probs_correct_recall[self.pd_col_name_threshold_prob][threshold]

    def get_hidden_neuron_spike_confidence_noisy_unknown(self, threshold):
        """
        Return the confidence that a hidden neuron will spike when the system is presented with a noisy memory
        that it has not learned given a specific threshold. This assumes a single hidden neuron.
        :param threshold:
        Hidden neuron threshold.
        :return:
        Confidence value
        """
        return self.probs_incorrect_recall[self.pd_col_name_threshold_prob][threshold]

    def get_probability_data_for_cache(self):
        """
        Generates a summary of the probability data.
        Unlike `get_prob_data_for_recall_types`, all the calculated probabilties are on the same row.
        This summary is a format conducive to caching. Each row also includes details of the scenario for
        cache search.
        :return:
        """
        summary_df = pd.DataFrame()
        summary_df[pd_col_name_t] = self.probs_perfect_recall[self.pd_col_name_activation_level]
        summary_df[pd_col_name_h_perfect_prob_act] = self.probs_perfect_recall[self.pd_col_name_activation_prob]
        summary_df[pd_col_name_h_correct_prob_act] = self.probs_correct_recall[self.pd_col_name_activation_prob]
        summary_df[pd_col_name_h_incorrect_prob_act] = self.probs_incorrect_recall[self.pd_col_name_activation_prob]
        summary_df[pd_col_name_h_perfect_prob_t] = self.probs_perfect_recall[self.pd_col_name_threshold_prob]
        summary_df[pd_col_name_h_correct_prob_t] = self.probs_correct_recall[self.pd_col_name_threshold_prob]
        summary_df[pd_col_name_h_incorrect_prob_t] = self.probs_incorrect_recall[self.pd_col_name_threshold_prob]

        # Remove the dataframe rows where the perfect probs are zero - these thresholds are too high
        summary_df = summary_df.loc[summary_df[pd_col_name_h_perfect_prob_t] != 0]
        num_rows = summary_df.shape[0]

        if num_rows == 0:
            print('Error: All threshold probabilities were zero. This is probably because there are no connections to '
                  'any hidden neurons with the current network definition. Try increasing the feature to hidden neuron '
                  'connection sparsity.')
            raise ValueError('Insufficient connections for problem space')

        summary_df.insert(0, pd_col_name_f_h_conn_type,
                          (np.full(shape=(num_rows,), fill_value=self.net_param_f_h_conn_type)))
        summary_df.insert(1, pd_col_name_f, (np.full(shape=(num_rows,), fill_value=self.net_param_f)))
        summary_df.insert(2, pd_col_name_h, (np.full(shape=(num_rows,), fill_value=self.net_param_h)))
        summary_df.insert(3, pd_col_name_f_h, (np.full(shape=(num_rows,), fill_value=self.net_param_f_h_sparsity)))
        summary_df.insert(4, pd_col_name_s, (np.full(shape=(num_rows,), fill_value=self.data_param_s_m)))
        summary_df.insert(5, pd_col_name_n, (np.full(shape=(num_rows,), fill_value=self.data_param_s_n)))

        return summary_df

    def get_prob_data_for_recall_types(self,
                                       include_perfect: bool = False,
                                       ) -> pd.DataFrame:
        """
        Returns the pandas dataframe containing the probability data. This is in a form conducive to highlighting
        all the different recall types, so each type of data for a particular scenario (perfect, correct, incorrect
        recall) has its own dedicated separate row.

        The returned dataframe has the following previously generated columns:
            * 'Activation or Threshold' - Integer value which indicates the activation / threshold to which the
               probability elements in the subsequent columns refer.
            * 'Activation Probability' - The probability that the neuron will activate at exactly the value
               specified in the 'Activation or Threshold' column.
            * 'Threshold Probability' - The probability that the neuron will fire assuming the threshold value
               specified in the 'Activation or Threshold' column.
            * 'Data Presented to Hidden Neuron' - static str value based on the data_type parameter that defines the
               type of data - 'Perfect memory recall', 'Correct recall', or 'Incorrect recall'.

        :param include_perfect:
        False if the perfect recall (correct with no noise) is not required.
        :return:
        Plot data in a dataframe.
        """
        if include_perfect:
            recall_type_data = pd.concat(
                [self.probs_correct_recall, self.probs_incorrect_recall, self.probs_perfect_recall], ignore_index=True)
        else:
            recall_type_data = pd.concat(
                [self.probs_correct_recall, self.probs_incorrect_recall], ignore_index=True)

        return recall_type_data

    def prob_correct_hoff(self,
                          m: int,
                          prob_h_firing_correct: float,
                          prob_h_firing_incorrect: float,
                          inhibition: bool):
        prob = 0
        incorrect_h = (m - 1) * self.net_param_h
        if inhibition:
            for num_h_firing in range(1, self.net_param_h + 1):
                prob_correct_h = n_choose_k_with_prob(self.net_param_h, num_h_firing, prob_h_firing_correct)
                prob_incorrect_fewer_h = n_choose_fewer_than_k_with_prob(incorrect_h,
                                                                         num_h_firing,
                                                                         prob_h_firing_incorrect)
                prob += prob_correct_h * prob_incorrect_fewer_h
        else:
            prob = (1 - (1 - prob_h_firing_correct) ** self.net_param_h) * (1 - prob_h_firing_incorrect) ** incorrect_h

        return prob

    def get_best_threshold(self, m: int):
        data = self.get_prob_data_for_accuracy(m)
        best_row = data['accuracy_inhib'].idxmax()
        recommended_threshold = data['threshold'][best_row]
        print('Recommended Threshold Accuracy inhib: ',  recommended_threshold)
        # For when f_h connections are treated as probability rather than proportion only:
        #
        # 'h_perfect_prob' gives the probability that any hidden neuron will have the same or more connections than
        # the threshold. Any falling outside this range will have too few connections and will be essentially
        # redundant. Therefore, to ensure an average of net_param_h hidden neurons that are not redundant,
        # the number of hidden neurons will require scaling accordingly.
        recommended_net_param_h = int(self.net_param_h / data['h_perfect_prob'][best_row])
        return recommended_threshold, recommended_net_param_h

    def get_prob_data_for_accuracy(self,
                                   m: int
                                   ) -> pd.DataFrame:
        """

        :param m:
            Number of memories
        :return:
        """
        pd_col_name_t = 'threshold'
        pd_col_name_h_perfect_prob = 'h_perfect_prob'
        pd_col_name_h_correct_prob = 'h_correct_prob'
        pd_col_name_h_incorrect_prob = 'h_incorrect_prob'

        accuracy_df = pd.DataFrame()
        accuracy_df[pd_col_name_t] = self.probs_perfect_recall[self.pd_col_name_activation_level]
        accuracy_df[pd_col_name_h_perfect_prob] = self.probs_perfect_recall[self.pd_col_name_threshold_prob]
        accuracy_df[pd_col_name_h_correct_prob] = self.probs_correct_recall[self.pd_col_name_threshold_prob]
        accuracy_df[pd_col_name_h_incorrect_prob] = self.probs_incorrect_recall[self.pd_col_name_threshold_prob]

        accuracy_df['accuracy_inhib'] = accuracy_df.apply(lambda row:
                                                          self.prob_correct_hoff(m=m,
                                                                                 prob_h_firing_correct=row[pd_col_name_h_correct_prob],
                                                                                 prob_h_firing_incorrect=row[pd_col_name_h_incorrect_prob],
                                                                                 inhibition=True), axis=1
                                                          )
        accuracy_df['accuracy_noinhib'] = accuracy_df.apply(lambda row:
                                                            self.prob_correct_hoff(m=m,
                                                                                   prob_h_firing_correct=row[pd_col_name_h_correct_prob],
                                                                                   prob_h_firing_incorrect=row[pd_col_name_h_incorrect_prob],
                                                                                   inhibition=False), axis=1)
        # print(accuracy_df.to_markdown())
        return accuracy_df

    def __calc_hidden_neuron_activation_probabilities(self,
                                                      probability: float,
                                                      data_type: str,
                                                      ) -> pd.DataFrame:
        """
        Create a dataframe containing the activation probabilities for a hidden neuron for signals characterised based
        on the parameters passed.
        The returned dataframe has the following columns:
              * 'Activation or Threshold' - Integer value which indicates the activation / threshold to which the
                probability elements in the subsequent columns refer.
              * 'Activation Probability' - The probability that the neuron will activate at exactly the value
                specified in the 'Activation or Threshold' column.
              * 'Threshold Probability' - The probability that the neuron will fire assuming the threshold value
                specified in the 'Activation or Threshold' column.
              * 'Data Presented to Hidden Neuron' - static str value based on the data_type parameter that defines the
                type of data - 'Perfect memory recall', 'Correct recall', or 'Incorrect recall'.
        :param probability:
            The probability that a learned f-h connection will be activated by the signal. If the signal is a memory
            that the hidden neuron has learned, then the probability will be 1. Otherwise, it will be something less
            than 1.
        :param data_type:
            The type of the signal being passed to the hidden neuron. This is populates the 'data_type' column.
        :return:
            Pandas dataframe containing the information required for this method.
        """

        prob_learned_connections = np.full(shape=(self.net_param_f + 1,), fill_value=0.0)

        if get_connection_type(self.net_param_f_h_conn_type) == ConnectionType.FIXED_PROBABILITY:
            # Connections are allocated according to a probability so will vary from hidden neuron to hidden neuron.
            # Step through all the possible number of learnt connections to a hidden neuron - 0-f
            # Establish the probability of that number of connections being learnt
            for num_learned_connections in range(len(prob_learned_connections)):
                prob_learned_connections[num_learned_connections] = \
                    n_choose_k_with_prob(self.net_param_f,
                                         num_learned_connections,
                                         self.f_h_learned_connection_prob)
        else:
            # The number of connections remains static for all hidden neurons (FIXED_NUMBER_PRE)
            # or for all feature neurons (FIXED_NUMBER_POST)
            # This is far simpler - Probability will always be 0 except for one value which will be 1.
            num_learned_connections = round(self.net_param_f * self.f_h_learned_connection_prob)
            prob_learned_connections[num_learned_connections] = 1

        # Calculate the activation probabilities
        activation_probabilities = np.full(shape=(self.net_param_f + 1,), fill_value=0.0)
        prob_so_far_for_activation = 0.0
        activation_value = 0
        probs_decreasing = False

        # First just consider the probability of an exact activation value, stepping through each case.
        # We pop these evaluations into a numpy array so that they can be used when summing for the thresholds.
        # This is more efficient than re-calculating for every threshold.
        #
        # The conditions of this while loop are constructed to ensure that the calculation
        # stops when the activation is so high that the probability is zero. This is for
        # efficiency.
        while (not (probs_decreasing & (prob_so_far_for_activation == 0.0))) \
                & (activation_value <= self.net_param_f):

            last_prob_so_far_for_activation = prob_so_far_for_activation
            prob_so_far_for_activation = 0

            for num_learned_connections in range(activation_value, self.net_param_f):
                prob_num_activations = n_choose_k_with_prob(num_learned_connections,
                                                            activation_value,
                                                            probability)
                prob_so_far_for_activation += prob_learned_connections[num_learned_connections] * \
                                              prob_num_activations

            activation_probabilities[activation_value] = round(prob_so_far_for_activation, 8)
            if self.debug:
                print('Activation value: {} has probability of {}'.format(activation_value,
                                                                          activation_probabilities[activation_value]))
            activation_value += 1
            if last_prob_so_far_for_activation > prob_so_far_for_activation:
                probs_decreasing = True

        # Derive the threshold probabilities from the activation probabilities
        threshold_probabilities = np.full(shape=activation_probabilities.shape, fill_value=0.0)
        total_prob_so_far = 0

        for threshold_value in range(len(activation_probabilities)-1, 0, -1):
            total_prob_so_far += activation_probabilities[threshold_value]
            threshold_probabilities[threshold_value] = round(total_prob_so_far, 8)

            if self.debug:
                print('Threshold value: {} has probability of {}'.format(threshold_value,
                                                                         threshold_probabilities[threshold_value]))

        if self.debug:
            print('Calculating threshold probabilities given activation probability of {}'.format(probability))
            print('Activation probabilities: \n', activation_probabilities)
            print('Threshold probabilities: \n', threshold_probabilities)
        # The probabilities will always be between 0 and 1. However, due to potential rounding up, occasionally the
        # threshold goes above 1 (e.g. 1.0000000001). This causes NaNs in later calculations, so clip the threshold
        # here.
        threshold_probabilities = np.clip(threshold_probabilities, 0, 1)

        # Put the derived the data in a dataframe
        df_out = self.generate_h_probs_df(
                                activations=np.arange(0, self.net_param_f + 1),
                                activation_probabilities=activation_probabilities,
                                threshold_probabilities=threshold_probabilities,
                                recall_type=data_type)

        return df_out


def generate_hidden_neuron_threshold(problem_space: dict,
                                     net_param_h: int,
                                     net_param_f_h_sparsity: float,
                                     net_param_f_h_conn_type: str,
                                     debug: bool = False,
                                     ):
    """
    Given the network and the problem space described by the input parameters, return the optimum threshold for the
    hidden neurons and the number of hidden neurons that need to be allocated per memory to cater for redundancy
    to achieve the required number of active (non-redundant) hidden neurons per memory.
    A redundant hidden neuron
    is one with fewer connections than the threshold - so will never be activated. An implementation could
    re-cycle these redundant hidden neurons for efficiency.
    Because this method is expensive for many tests, we cache the probability results used to calculate the
    optimum threshold in a file.
    :param problem_space:
    :param net_param_h:
    :param net_param_f_h_sparsity:
    :param debug:
    :param cache_file_dir:
    :param cache_file_name:
    :param cache_file_sheet:
    :param save_data:
    Save data in the cache file. This will only happen if the data wasn't already in the cache (to avoid duplication).
    :return:
    """

    print('Establishing hidden neuron threshold for the following test:'
          '\n    net_param_f:             {}'
          '\n    net_param_h:             {}'
          '\n    net_param_f_h_sparsity:  {}'
          '\n    net_param_f_h_conn_type: {}'
          '\n    data_param_s_m:          {}'
          '\n    data_param_s_n:          {}'
          '\n    data_param_m:            {}'
          .format(problem_space['f'], net_param_h, net_param_f_h_sparsity, net_param_f_h_conn_type, problem_space['s_m'],
                  problem_space['s_n'], problem_space['m']))

    hp_generator = HiddenNeuronParameters(problem_space=problem_space,
                                              net_param_h=net_param_h,
                                              net_param_f_h_sparsity=net_param_f_h_sparsity,
                                              net_param_f_h_conn_type=net_param_f_h_conn_type,
                                              debug=debug
                                              )

    recommended_threshold, recommended_net_param_h = hp_generator.get_best_threshold(m=problem_space['m'])

    av_conns_per_h = int(problem_space['f'] * problem_space['s_m'] * net_param_f_h_sparsity)

    print('    The recommended hidden neuron threshold is {}'.format(recommended_threshold))
    if net_param_f_h_conn_type != 'FixedProbability':
        print('    Using static f-h connections proportion: {} gives exactly {} connections per hidden neuron.'
              .format(net_param_f_h_sparsity, av_conns_per_h))
        if av_conns_per_h < recommended_threshold:
            required_sparsity = recommended_threshold / (problem_space['f'] * problem_space['s_m'])
            print('    WARNING: The hidden neurons do not have sufficient connections to fire.\n'
                  '             Consider increasing the f_h sparsity to at least {}.'.format(required_sparsity))
    else:
        print('    Using probability of f-h connections per hidden neuron: {} gives an average of : {} connections\n'
              .format(net_param_f_h_sparsity, av_conns_per_h))
        if recommended_net_param_h > net_param_h:
            print('    Warning: {} hidden neurons have been allocated per memory, but some will have insufficient '
                  'connections to ever fire (they will be redundant).\n'
                  '             To ensure {} non-redundant hidden neurons, '
                  'consider increasing the f_h sparsity or increase the number of hidden neurons for allocated'
                  'to each memory to {}.'.format(net_param_h, net_param_h, recommended_net_param_h))

    return recommended_threshold, recommended_net_param_h
