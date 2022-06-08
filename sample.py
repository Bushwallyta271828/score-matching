import numpy as np
from numpy.random import normal
from numpy.random import uniform
import sufficient_statistics


def sample(theta_star, samples):
    current_samples = np.array([])
    while np.size(current_samples) < samples:
        #I could create more samples at a time
        #if most of them looked like they would get rejected,
        #but this is a good enough baseline.
        sigma = theta_star[0]**(-0.5)
        xs = normal(loc=0.0, scale=sigma, size=samples)
        log_prob_accept = sufficient_statistics.later_sufficient_statistics(xs, theta_star[1:])
        prob_accept = np.exp(log_prob_accept)
        #raise error if any prob_accept is greater than 1
        if np.any(prob_accept > 1):
            raise ValueError("prob_accept > 1; check sufficient_statistics.later_sufficient_statistics is negative")
        #accept xs with probability prob_accept
        random_values = uniform(size=samples)
        accepted_xs = xs[random_values < prob_accept]
        current_samples = np.append(current_samples, accepted_xs)
    return current_samples[:samples]
