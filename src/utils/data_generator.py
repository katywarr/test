import numpy as np
import math
from utils.similarity_scores import compare_tests_to_memories, jaccard_similarity


def generate_signals(num_signals: int, f: int, active_features: int):
    """
    Randomly generate a set of binary signals, with the same number of active features.
    :param num_signals:
    Number of signals to generate
    :param f:
    Number of feature neurons on which to project the signals (the message dimensionality)
    :param active_features:
    The length of each signal. This is the number of values that will be set to 1 in the returned array
    representing the signal.
    :return:
    Numpy array of shape (num_signals, f_neurons) where the 1s represent part of the signal.
    """
    if active_features > f:
        print('Error: Unable to generate signals of length {}. Insufficient neurons available: {}'
              .format(active_features, f))
        return None
    signals = np.full(shape=(num_signals, f), fill_value=0)
    signals[:, 0:active_features] = 1
    np.apply_along_axis(np.random.shuffle, axis=1, arr=signals)
    return signals


def generate_signals_prob_length(num_signals: int, f_neurons: int, sparsity_prob: float):
    """
    Randomly generate a set of binary signals based on a probability of each feature being active in the signal. The
    number of active features in each signal will vary according to poisson distribution.
    :param num_signals:
    Number of signals to generate
    :param f_neurons:
    Number of feature neurons on which to project the signals (the message dimensionality)
    :param sparsity_prob:
    The probability of any feature value being active in a signal. This is the probability of each value being set to 1.
    :return:
    Numpy array of shape (num_signals, f_neurons) where the 1s represent part of the signal.
    """
    signals = np.random.choice(np.array([0, 1]), size=(num_signals, f_neurons), p=[1-sparsity_prob, sparsity_prob])
    return signals


def generate_test_data_for_memory(memory: np.ndarray,
                                  bits_to_flip: int,
                                  num_test_signals):
    """
    Randomly flip a defined number of bits in a vector representing a memory to create a set of test signals that
    are occluded versions of that memory.
    :param memory:
    :param bits_to_flip:
    :param num_test_signals:
    :return:
    """
    test_signals_no_noise = np.repeat([memory], repeats=num_test_signals, axis=0)
    bit_flip_array = np.zeros(shape=test_signals_no_noise.shape)
    bit_flip_array[:, 0:bits_to_flip] = 1
    for row in bit_flip_array:
        np.random.shuffle(row)
    return np.logical_xor(test_signals_no_noise, bit_flip_array)


def generate_test_data_for_memories(memories: np.ndarray,
                                    bits_to_flip: int,
                                    num_test_signals_per_memory: int):
    """
    Randomly flip a defined number bits in a vectors representing a memories to create a sets of test signals for
    each memory.
    :param memories:
    :param bits_to_flip:
    :param num_test_signals_per_memory:
    :return:
    """

    num_memories = memories.shape[0]
    num_features = memories.shape[1]
    num_test_signals = num_memories * num_test_signals_per_memory
    test_signals = np.empty(shape=(num_test_signals, num_features))
    ground_truth_signals = np.empty(shape=(num_test_signals, num_features))
    ground_truth_labels = np.full(shape=(num_test_signals,), fill_value=-1)

    memory_start_index = 0

    for i in range(num_memories):
        # For each memory, create a set of test signals and a corresponding ground_truth array
        memory_end_index = num_test_signals_per_memory * (i+1)
        test_signals[memory_start_index:memory_end_index] = \
            generate_test_data_for_memory(
            memories[i],
            bits_to_flip=bits_to_flip,
            num_test_signals=num_test_signals_per_memory)
        ground_truth_signals[memory_start_index:memory_end_index] = \
            np.repeat([memories[i]], repeats=num_test_signals_per_memory, axis=0)
        ground_truth_labels[memory_start_index:memory_end_index] = \
            np.full(shape=(num_test_signals_per_memory,), fill_value=i)
        memory_start_index = memory_end_index
    return test_signals, ground_truth_labels


def get_highest_similarity_index(similarities: np.ndarray) -> (int, bool):
    selected_nearest_memory = int(np.argmax(similarities))
    # It's incredibly rare to have more than one ground truth with exactly the same Jaccard similarity.
    # Default to the first one, but print out a warning.
    competing_gts = np.argwhere(similarities ==
                                similarities[selected_nearest_memory]).flatten()
    num_competing = max(competing_gts.shape[0] - 1, 0)
    if competing_gts.shape[0] == similarities.shape[0]:
        # All the similarities were exactly the same (probably zero)
        selected_nearest_memory = -1
        print('INFO: No closest memories found for signal. \n'
              '      Closest was set to {}. All memories had the signal similarity measure: {}'
              .format(selected_nearest_memory, similarities[selected_nearest_memory]))
    else:
        if num_competing != 0:
            # Randomly select one of the possible options if there is more than one
            selected_nearest_memory = np.random.choice(competing_gts, 1)
            print('INFO: More than closest memory was found for signal. \n'
                  '      Closest {} was selected from {}. All had the signal similarity measure: {}'
                  .format(selected_nearest_memory, competing_gts, similarities[selected_nearest_memory]))

    return selected_nearest_memory, num_competing


def get_nearest_memory(signal, memories) -> (int, int):
    """
    Find the memory that a signal is closest to (using Jaccard similarity measurement).
    If more than one memory has the highest similarity measurement, an information message is printed and only one
    value is returned along with an integer indicating the number of competing memories.
    :param signal:
    Signal to be compared.
    :param memories:
    Numpy array of memories with the same length as the signal.
    :return:
    The index of the memory to which the signal is closest to and the number of other memories at the same distance
    (usually this is zero)
    """
    jaccard_differences_per_mem = compare_tests_to_memories(test_signals=np.array([signal]),
                                                            memories=memories,
                                                            comparison_function=jaccard_similarity)[0]

    return get_highest_similarity_index(jaccard_differences_per_mem)


def get_nearest_memories(signals, memories) -> np.ndarray:
    nearest_memories = np.empty(shape=(signals.shape[0],), dtype=int)
    # This could be done more efficiently but get_ground_truth_label will display a message when there is more
    # than one possibility.
    for index in range(signals.shape[0]):
        nearest_memories[index], _ = get_nearest_memory(signal=signals[index], memories=memories)
    return nearest_memories


def generate_simulation_data(problem_space: dict,
                             num_simulations: int,
                             ):
    """
    Generate the data for a set of simulations. Return memory signals and recall signals with their appropriate ground
    truth memory labels.

    Simulation data is generated according to the problem space characterisation, which is passed as a dictionary:

    * ``f`` - Signal length (number of features).
    * `'m`` - The number of memories to learn.
    * ``s_m`` - Memory signal sparsity. The proportion of features that are ones in the signal. The number of ones in
        each memory signal is ``f * s_m``. The remainder of the features are set to zero.
    * ``s_n`` - Noise sparsity. The proportion of features in a memory signal that are flipped (one to zero or
        zero to one) to generate a recall signal.
    * ``s_is_prob`` - Set to True if the sparsity values should be treated as probabilities rather than proportions.
        This dictionary parameter is optional, the default is False. Setting to True might be more representative of
        real life applications, but False is better for theoretical examination

    :param problem_space: A dictionary containing the parameters describing the problem space, as defined above.
    :param num_simulations: Number of simulations to generate data for. This defines how many recall signals
        will be generated.
    :return: `data_param_m`` recall signals each of length ``data_param_f``,
        ``num_simulations`` recall signals each of length ``data_param_f``,
        ``num_simulations`` integer ground truths identifying the memories from which each of the recall signals
        originated.
    """

    print('======= Generating Simulation Data ======='
          '\n      Number of features (feature neurons): {}'
          '\n      Number of memories:                   {}'
          '\n      Number of simulations:                {}'
          .format(problem_space['f'], problem_space['m'], num_simulations))

    # Ensure that there are sufficient tests per memory to cater for the number of tests to run.
    num_tests_per_memory = math.ceil(num_simulations / problem_space['m'])

    signal_length = int(problem_space['f'] * problem_space['s_m'])
    bits_to_flip = int(problem_space['f'] * problem_space['s_n'])
    print('      Signal length:                        {}'
          '\n      Bits to flip for noise:               {}'
          .format(signal_length, bits_to_flip))
    memory_signals = generate_signals(num_signals=problem_space['m'],
                                      f=problem_space['f'],
                                      active_features=signal_length)

    test_signals, ground_truth_labels = generate_test_data_for_memories(
        memories=memory_signals,
        bits_to_flip=bits_to_flip,
        num_test_signals_per_memory=num_tests_per_memory)

    return memory_signals, test_signals[0:num_simulations], ground_truth_labels[0:num_simulations]
