import numpy as np
import math
import test_class
from mle_accuracy import mle
from scorematching_accuracy import scorematching
from limit_covariance import mle_limit_covariance
from limit_covariance import scorematching_limit_covariance
from file_read_write import write_to_file
import sufficient_statistics


def parameters_to_results(test_parameters):
    #test_parameters is of type test_class.TestParameters
    if test_parameters.method in ['mle', 'scorematching']:
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
        results = test_class.TestResults(accuracy=sum(accuracies)/len(accuracies), mean=mu, cov=sigma)
    elif test_parameters.method in ['mle_limit', 'scorematching_limit']:
        print("method:", test_parameters.method,
                "suffstats:", [str(suffstat) for suffstat in test_parameters.suffstats],
                "n:", test_parameters.n,
                "theta_star:", test_parameters.theta_star)
        if test_parameters.method == 'mle_limit':
            cov = mle_limit_covariance(test_parameters)
        elif test_parameters.method == 'scorematching_limit':
            cov = scorematching_limit_covariance(test_parameters)
        mu = test_parameters.theta_star
        #Create test_parameters.n samples from a multivariate Gaussian
        #with mean 0 and covariance cov:
        samples = np.random.multivariate_normal(np.zeros(len(test_parameters.suffstats)), cov, test_parameters.runs)
        accuracies = np.sqrt(np.sum(samples**2, axis=1))
        results = test_class.TestResults(accuracy=np.mean(accuracies), mean=mu, cov=cov)
    else:
        raise ValueError('unrecognized method')
    return results


def run_tests(parameters_for_tests):
    #parameters_for_tests is a list of test_class.TestParameters
    #returns a list of test_class.Test
    tests = []
    for test_parameters in parameters_for_tests:
        test_results = parameters_to_results(test_parameters)
        test = test_class.Test(test_parameters, test_results)
        tests.append(test)
    return tests



#Old old code:
#def generate_data_log_spacing(n_start, n_stop, num_ns, runs):
#    ns = np.exp(np.linspace(math.log(n_start), math.log(n_stop), num=num_ns))
#    ns = ns.astype(int)
#    mle_accuracies = np.array([mle_average(n, runs) for n in ns])
#    scorematch_accuracies = np.array([scorematching_average(n, runs) for n in ns])
#    return (ns, mle_accuracies, scorematch_accuracies)


#def generate_data_changing_theta_1(n, runs, theta_1_start, theta_1_stop, num_theta_1s):
#    theta_1s = np.linspace(theta_1_start, theta_1_stop, num=num_theta_1s)
#    thetas = np.array([np.ones(num_theta_1s), theta_1s]).T

#Old code:
#def generate_data_changing_theta_1(n, runs, theta_1_start, theta_1_stop, num_theta_1s):
#    methods = ['mle']*num_theta_1s + ['scorematching']*num_theta_1s
#    ns = [n]*num_theta_1s*2
#    runs = [runs]*num_theta_1s*2
#    theta_1_range = np.exp(np.linspace(math.log(theta_1_start), math.log(theta_1_stop), num=num_theta_1s))
#    theta_star_range = [np.array([1.0, theta_1]) for theta_1 in theta_1_range]
#    theta_stars = theta_star_range + theta_star_range
#    accuracies = []
#    means = []
#    covs = []
#    for i in range(len(methods)):
#        output = aggregate(methods[i], ns[i], runs[i], theta_stars[i])
#        accuracies.append(output[0])
#        means.append(output[1][0])
#        covs.append(output[1][1])
#    return (methods, ns, runs, theta_stars, accuracies, means, covs)


def changing_exponent_parameters():
    #collect inputs from user:
    n = int(input("Enter n: "))
    runs = int(input("Enter runs: "))
    exponent_start = float(input("Enter exponent start: "))
    exponent_stop = float(input("Enter exponent stop: "))
    num_exponents = int(input("Enter number of exponents: "))
    
    #generate parameters:
    exponents = np.exp(np.linspace(math.log(exponent_start), math.log(exponent_stop), num=num_exponents))
    exponents = 2 * (exponents / 2).astype(int)
    exponents = np.maximum(exponents, 4)

    parameters_for_tests = []
    for method in ['mle', 'scorematching']:
        for exponent in exponents:        
            suffstats = [sufficient_statistics.FirstStat(), sufficient_statistics.PolyStat(exponent)]
            parameters_for_tests.append(test_class.TestParameters(suffstats, np.array([1.0, 1.0]), n, method, runs))
    return parameters_for_tests


def asymptotic_test_parameters():
    exponent = int(input("Enter exponent: "))
    runs = int(input("Enter runs: "))
    start_n = int(input("Enter start n: "))
    stop_n = int(input("Enter stop n: "))
    num_ns = int(input("Enter number of ns: "))

    ns = np.exp(np.linspace(math.log(start_n), math.log(stop_n), num=num_ns))
    ns = ns.astype(int)

    parameters_for_tests = []
    for method in ['mle', 'scorematching']:
        for n in ns:
            suffstats = [sufficient_statistics.FirstStat(), sufficient_statistics.PolyStat(exponent)]
            parameters_for_tests.append(test_class.TestParameters(suffstats, np.array([1.0, 1.0]), n, method, runs))
    return parameters_for_tests


#n_start = 2000
#n_stop = 20000
#num_ns = 5
#runs = 10000
#output = generate_data_log_spacing(n_start=n_start, n_stop=n_stop, num_ns=num_ns, runs=runs)
#ns, mle_accuracies, scorematch_accuracies = output
#name = './results/log_data_'
#name += 'n_start_' + str(n_start)
#name += '_n_stop_' + str(n_stop)
#name += '_num_ns_' + str(num_ns)
#name += '_runs_' + str(runs)
#name += '.txt'
#write_to_file(ns, mle_accuracies, scorematch_accuracies, name=name)

#methods = ["mle",]*5 + ["scorematching",]*5
#ns = [2000,]*10
#runs = [100,]*10
#theta_stars = [np.array([1.0, 0.5]),
#                np.array([1.0, 0.75]),
#                np.array([1.0, 1.0]),
#                np.array([1.0, 1.25]),
#                np.array([1.0, 1.5]),
#                np.array([1.0, 0.5]),
#                np.array([1.0, 0.75]),
#                np.array([1.0, 1.0]),
#                np.array([1.0, 1.25]),
#                np.array([1.0, 1.5])]

#Old code:
##read from user input
#n = int(input("n: "))
#runs = int(input("runs: "))
#theta_1_start = float(input("theta_1_start: "))
#theta_1_stop = float(input("theta_1_stop: "))
#num_theta_1s = int(input("num_theta_1s: "))
#output = generate_data_changing_theta_1(n=n, runs=runs, theta_1_start=theta_1_start, theta_1_stop=theta_1_stop, num_theta_1s=num_theta_1s)
#methods, ns, runs, theta_stars, accuracies, means, covs = output
#write_to_file(methods, ns, runs, theta_stars, accuracies, means, covs)

##New code:
##read from user input
#n = int(input("n: "))
#runs = int(input("runs: "))
#exponent_start = float(input("exponent_start: "))
#exponent_stop = float(input("exponent_stop: "))
#num_exponents = int(input("num_exponents: "))
#tests = generate_data_changing_exponent(n=n, runs=runs, exponent_start=exponent_start, exponent_stop=exponent_stop, num_exponents=num_exponents)
#write_to_file(tests)

parameters_for_tests = asymptotic_test_parameters()
tests = run_tests(parameters_for_tests)
write_to_file(tests)
