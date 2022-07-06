import numpy as np
import math
import test_class
from mle_accuracy import mle
from scorematching_accuracy import scorematching
from limit_covariance import mle_limit_covariance
from limit_covariance import scorematching_limit_covariance
import sufficient_statistics


def conventional_test(test_parameters):
    accuracies = []
    displacements = []
    for i in range(test_parameters.runs):
        print("method:", test_parameters.method,
                "suffstats:", [str(suffstat) for suffstat in test_parameters.suffstats],
                "n:", test_parameters.n,
                "theta_star:", test_parameters.theta_star,
                "run:", i)
        if test_parameters.method == 'mle':
            output = mle(test_parameters)
        elif test_parameters.method == 'scorematching':
            output = scorematching(test_parameters)
        displacements.append(output[0] - test_parameters.theta_star)
        accuracies.append(output[1])
    displacements = np.array(displacements)
    #fit 2d Gaussian to displacements
    mu = np.mean(displacements, axis=0)
    sigma = np.cov(displacements.T) #Note that this automatically subtracts out mean!
    return test_class.TestResults(accuracy=sum(accuracies)/len(accuracies), mean=mu, cov=sigma)


def limit_test(test_parameters):
    print("method:", test_parameters.method,
            "suffstats:", [str(suffstat) for suffstat in test_parameters.suffstats],
            "n:", test_parameters.n,
            "theta_star:", test_parameters.theta_star)
    if test_parameters.method == 'mle_limit':
        cov = mle_limit_covariance(test_parameters)
    elif test_parameters.method == 'scorematching_limit':
        cov = scorematching_limit_covariance(test_parameters)
    mu = np.zeros(len(test_parameters.suffstats))
    #Create test_parameters.runs samples from a multivariate Gaussian
    #with mean 0 and covariance cov:
    samples = np.random.multivariate_normal(mu, cov, test_parameters.runs)
    accuracies = np.sqrt(np.sum(samples**2, axis=1))
    return test_class.TestResults(accuracy=np.mean(accuracies), mean=mu, cov=cov)


def parameters_to_results(test_parameters):
    #test_parameters is of type test_class.TestParameters
    if test_parameters.method in ['mle', 'scorematching']:
        return conventional_test(test_parameters)
    elif test_parameters.method in ['mle_limit', 'scorematching_limit']:
        return limit_test(test_parameters)
    else:
        raise ValueError('unrecognized method')


def run_tests(parameters_for_tests):
    #parameters_for_tests is a list of test_class.TestParameters
    #returns a list of test_class.Test
    tests = []
    for test_parameters in parameters_for_tests:
        test_results = parameters_to_results(test_parameters)
        test = test_class.Test(test_parameters, test_results)
        tests.append(test)
    return tests

