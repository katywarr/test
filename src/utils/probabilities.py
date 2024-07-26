from scipy.stats import binom


def n_choose_k_with_prob(n, k, prob_k):
    """
    Returns the probability that, given k choices, each with prob_k of success, from n independent binomial
    possibilities, that all the choices will be successful.
    :param n:
    The total number to select from.
    :param k:
    The number of selections to make.
    :param prob_k:
    The probability that a single selection from n is successful.
    :return:
    The confidence (probability) that a selection k is successful.
    """
    # print('debug: ', n, k, prob_k )
    # return math.comb(n, k) * (prob_k ** k) * ((1 - prob_k) ** (n - k))
    # prob = math.comb(n, k) * (prob_k ** k) * ((1 - prob_k) ** (n - k))
    # Using the probability mass function provided by scipi
    # has the advantage that it deals with larger values
    prob = binom.pmf(k, n, prob_k)
    return prob


def n_choose_fewer_than_k_with_prob(n, k, prob):
    """
    Sums the probabilities that, given 0 to k-1 choices,
    each with prob of success, from n independent binomial
    possibilities, that all the choices will be successful.
    :param n:
    The total number to select from.
    :param k:
    The number of selections to make.
    :param prob_k:
    The probability that a single selection from n is successful.
    :return:
    The confidence (probability) that a selection k is successful.
    """
    return_prob = 0
    for select in range(0, k):
        return_prob += n_choose_k_with_prob(n, select, prob)
    return return_prob


def prob_f_h_carries_signal(recall_type:str, s_m: float, s_n: float):
    """
    The probability of a connection carrying a signal: $P_{c\_signal}({recall\_type})$
    Given a leant connection, `prob_conn_signal` describes the probability that the connection
    will carry a signal during recall. This varies depending on the recall type.
    :param recall_type:
    :param s_m:
    :param s_n:
    :return:
    """
    if recall_type == 'correct':
        # With a noisy memory, the probability that an f-h learned connection is activated is the prob of a
        # learned connection is 1 multiplied by (1 minus the probability that the learned signal contains noise)
        # which equates to 1-noise.
        prob = 1 - s_n
    else:
        if recall_type == 'incorrect':
            # With a memory unknown to the hu, the probability that an f-h learned connection is activated is the
            # probability that that bit of the recall signal is a 1.
            s_r = s_m * (1-s_n) + ((1-s_m) * s_n)      # Recall signal sparsity
            prob = s_r
        else:
            if recall_type == 'perfect_mem':
                # Shorthand for 'correct' when s_n == 0
                prob = 1
            else:
                raise ValueError('Recall type: '+recall_type+' is not valid.')

    # print('Probability of signal on a connection for recall type {} is: {}\n'.format(recall_type, prob))
    return prob


def prob_post_conns(num_conns: int,
                    pre_population_size: int,
                    conn_sparsity: float,
                    conn_type: str):
    """
    Returns the probability that a specific number of post synaptic connections will be established
    to a neuron from a sparse selection of pre-synatic population.


    :param num_conns:
        Number of connections that the return probability refers to.
    :param pre_population_size:
        Size of the pre-population
    :param conn_sparsity:
        The sparsity of connections from the pre-synaptic sub-set.
    :param conn_type:
        The connection rule used in establishing the connections:
        'FixedProbability' or FixedNumberPre' or FixedNumberPost'
    """
    prob = 0
    if (conn_type == 'FixedProbability') or (conn_type == 'FixedNumberPost'):
        prob = n_choose_k_with_prob(pre_population_size, num_conns, conn_sparsity)
    else:
        if conn_type == 'FixedNumberPre':
            static_c = round(pre_population_size * conn_sparsity)
            if num_conns == static_c:
                prob = 1
        else:
            raise ValueError('Connection type: ' + conn_type + ' is not valid.')
    # print('Probability of {} connections is: {}'.format(num_conns, prob))
    return prob


def prob_activation(a: int,
                    recall_type: str,
                    max_connections: int,
                    signal_fn,
                    connection_fn):
    prob = 0
    prob_signal = signal_fn(recall_type=recall_type)
    for num_conns in range(a, max_connections):
        prob_a_for_num_conns = n_choose_k_with_prob(num_conns, a, prob_signal)
        #print('     Probability of activation {} with {} connections is: {}'
        #      .format(a, num_conns, prob_a_for_num_conns))
        prob += connection_fn(num_conns) * prob_a_for_num_conns
    # print('Probability of activation {} is: {}'.format(a, prob))
    return prob


def prob_firing(recall_type: str, theta: int, max_conns: int, activation_fn):
    """
    The probability of a hidden neuron firing $P_{h\_fires}({recall\_type})$
    The probability of a hidden neuron firing when presented with a recall signal is defined by the `prob_firing` function.
    :param recall_type:
    :param theta:
    :param max_conns:
    :param activation_fn:
    :return:
    """
    prob = 0
    for activation in range(theta, max_conns):
        prob += activation_fn(activation, recall_type)
    return prob


def prob_correct_hoff(m: int, h: int,
                      prob_h_firing_correct: float,
                      prob_h_firing_incorrect: float,
                      inhibition: bool):
    prob = 0
    incorrect_h = (m-1)*h
    if inhibition:
        for num_h_firing in range(1, int(h)+1):
            prob_correct_h = n_choose_k_with_prob(h, num_h_firing, prob_h_firing_correct)
            prob_incorrect_fewer_h = n_choose_fewer_than_k_with_prob(incorrect_h,
                                                                     num_h_firing,
                                                                     prob_h_firing_incorrect)
            prob += prob_correct_h * prob_incorrect_fewer_h
    else:
        prob = (1-(1-prob_h_firing_correct)**h) * (1-prob_h_firing_incorrect)**incorrect_h
    return prob


"""
Lambdas for re-use.
The following functions return lambdas that can be re-used for a specific scenario.
"""


def prob_f_h_carries_signal_fn(s_m: float, s_n: float):
    """
    Return a prob_f_h_carries_signal lambda that can be re-used for the problem space across different recall types
    to determine the probability of a signal on a feature to hidden neuron connection.
    """
    return lambda recall_type: prob_f_h_carries_signal(recall_type=recall_type, s_m=s_m, s_n=s_n)


def prob_post_conns_fn(length, sparsity, conn_type):
    """
    Return a prob_post_conns function that can be re-used for a problem space and network.
    """
    return lambda c: prob_post_conns(num_conns=c, pre_population_size=length,
                                     conn_sparsity=sparsity, conn_type=conn_type)


def prob_activation_fn(max_connections, signal_fn, connection_fn):
    """
    Return a prob_activation function that can be re-used for a problem scenario and network with just the activation
    changing.
    """
    return lambda a, recall_type: prob_activation(a=a,
                                                  recall_type=recall_type,
                                                  max_connections=max_connections,
                                                  signal_fn=signal_fn,
                                                  connection_fn=connection_fn)


def prob_firing_fn(max_conns, activation_fn):
    """
    Return a prob_firing function that can be re-used for a problem scenario with just the recall type and
    value for theta changing.
    :param max_conns:
    :param activation_fn:
    :return:
    """
    return lambda recall_type, theta: prob_firing(recall_type=recall_type,
                                                  theta=theta,
                                                  max_conns=max_conns,
                                                  activation_fn=activation_fn)