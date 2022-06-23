import numpy as np
from numpy.random import normal
from numpy.random import uniform
import sufficient_statistics
import test_class


def sample(test_parameters):
    assert(np.all(test_parameters.theta_star >= 0))
    #test_parameters is of type test_class.TestParameters
    current_samples = np.array([])
    while np.size(current_samples) < test_parameters.n:
        #I could create more samples at a time
        #if most of them looked like they would get rejected,
        #but this is a good enough baseline.
        sigma = test_parameters.theta_star[0]**(-0.5)
        xs = normal(loc=0.0, scale=sigma, size=test_parameters.n)

        later_suff_stat_values = sufficient_statistics.zeroth_derivatives(test_parameters.suffstats[1:], xs)
        assert(np.all(later_suff_stat_values <= 0))
        log_prob_accept = np.dot(test_parameters.theta_star[1:], later_suff_stat_values)
        prob_accept = np.exp(log_prob_accept)
        assert(np.all(prob_accept <= 1))

        #accept xs with probability prob_accept
        random_values = uniform(size=test_parameters.n)
        accepted_xs = xs[random_values < prob_accept]
        current_samples = np.append(current_samples, accepted_xs)
    return current_samples[:test_parameters.n]
