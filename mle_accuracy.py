import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import sufficient_statistics
import sample
import test_class
#import matplotlib.pyplot as plt


def non_normalized_pdf(x, thetas, suffstats):
    suff_stat_values = sufficient_statistics.zeroth_derivatives(suffstats, np.array([x]))
    exponent = np.dot(thetas, suff_stat_values)[0]
    return np.exp(exponent)


def normalize(thetas, suffstats):
    #compute the partition function
    return quad(non_normalized_pdf, -np.infty, np.infty, args=(thetas, suffstats))[0]


def negative_log_likelihood(thetas, suffstats, xs):
    if np.any(thetas <= 0):
        print("WARNING: theta <= 0")
        #Note: sometimes minimize passes a theta with a negative element.
        return 10**6 * (-np.min(thetas) + 1) #gradient should push in right direction
    Z = normalize(thetas, suffstats)
    suff_stat_values = sufficient_statistics.zeroth_derivatives(suffstats, xs)
    non_normalized_exponents = np.dot(thetas, suff_stat_values)
    log_likelihood = np.sum(non_normalized_exponents - np.log(Z))
    return -log_likelihood


def mle(test_parameters):
    #test_parameters is of type test_class.TestParameters
    assert(test_parameters.method == "mle")
    assert(np.all(test_parameters.theta_star >= 0))
    samples = sample.sample(test_parameters)
    #find the MLE with theta bounded to be positive:
    bounds = [(0, None) for i in range(np.size(test_parameters.theta_star))]
    MLE_answer = minimize(negative_log_likelihood,
                            test_parameters.theta_star,
                            args=(test_parameters.suffstats, samples),
                            bounds=bounds).x
    dist = np.sqrt(np.sum((MLE_answer - test_parameters.theta_star)**2))
    return (MLE_answer, dist)


#xs = []
#ys = []
#for i in range(1,101):
#    n = 100*i
#    print("n =", n)
#    dist = mle_accuracy(n, np.array([1.5, 0.5]), np.array([1, 1]))[1]
#    xs.append(n)
#    ys.append(dist)
#plt.plot(xs, ys, '.')
#plt.show()
