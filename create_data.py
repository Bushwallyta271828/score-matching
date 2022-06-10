import numpy as np
import math
from mle_accuracy import mle_accuracy
from scorematching_accuracy import scorematching_accuracy
from file_read_write import write_to_file


def aggregate(method, n, runs, theta_star):
    accuracies = []
    displacements = []
    for i in range(runs):
        print("method:", method, "n:", n, "theta_star:", theta_star, "run:", i)
        if method == 'mle':
            output = mle_accuracy(n, theta_star=theta_star, initial_theta=theta_star)
            #Note: setting initial_theta to theta_star is questionable
            #but mle_accuracy seems to focus enough on actually maximizing the liklihood
            #that I trust it not to just exit on the first iteration or something.
        elif method == 'scorematching':
            output = scorematching_accuracy(n, theta_star=theta_star)
        else:
            raise ValueError('method must be either mle or scorematching')
        displacements.append(output[0] - theta_star)
        accuracies.append(output[1])
    displacements = np.array(displacements)
    #fit 2d Gaussian to displacements
    mu = np.mean(displacements, axis=0)
    sigma = np.cov(displacements.T)
    return sum(accuracies) / len(accuracies), (mu, sigma)


#Old code:
#def generate_data_log_spacing(n_start, n_stop, num_ns, runs):
#    ns = np.exp(np.linspace(math.log(n_start), math.log(n_stop), num=num_ns))
#    ns = ns.astype(int)
#    mle_accuracies = np.array([mle_average(n, runs) for n in ns])
#    scorematch_accuracies = np.array([scorematching_average(n, runs) for n in ns])
#    return (ns, mle_accuracies, scorematch_accuracies)


#def generate_data_changing_theta_1(n, runs, theta_1_start, theta_1_stop, num_theta_1s):
#    theta_1s = np.linspace(theta_1_start, theta_1_stop, num=num_theta_1s)
#    thetas = np.array([np.ones(num_theta_1s), theta_1s]).T


def generate_data_changing_theta_1(n, runs, theta_1_start, theta_1_stop, num_theta_1s):
    methods = ['mle']*num_theta_1s + ['scorematching']*num_theta_1s
    ns = [n]*num_theta_1s*2
    runs = [runs]*num_theta_1s*2
    theta_1_range = np.exp(np.linspace(math.log(theta_1_start), math.log(theta_1_stop), num=num_theta_1s))
    theta_star_range = [np.array([1.0, theta_1]) for theta_1 in theta_1_range]
    theta_stars = theta_star_range + theta_star_range
    accuracies = []
    means = []
    covs = []
    for i in range(len(methods)):
        output = aggregate(methods[i], ns[i], runs[i], theta_stars[i])
        accuracies.append(output[0])
        means.append(output[1][0])
        covs.append(output[1][1])
    return (methods, ns, runs, theta_stars, accuracies, means, covs)




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

#read from user input
n = int(input("n: "))
runs = int(input("runs: "))
theta_1_start = float(input("theta_1_start: "))
theta_1_stop = float(input("theta_1_stop: "))
num_theta_1s = int(input("num_theta_1s: "))
output = generate_data_changing_theta_1(n=n, runs=runs, theta_1_start=theta_1_start, theta_1_stop=theta_1_stop, num_theta_1s=num_theta_1s)
methods, ns, runs, theta_stars, accuracies, means, covs = output
write_to_file(methods, ns, runs, theta_stars, accuracies, means, covs)
