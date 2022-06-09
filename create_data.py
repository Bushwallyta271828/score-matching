from mle_accuracy import mle_accuracy
from scorematching_accuracy import scorematching_accuracy
import numpy as np
import math


def mle_average(n, runs):
    """
    Calculates the average MLE accuracy over a number of runs
    n: sample size per run
    runs: number of runs
    """
    mle_accuracies = []
    for i in range(runs):
        print("method: mle_average", "n:", n, "run:", i)
        mle_accuracies.append(mle_accuracy(n, theta_star=np.array([1.5, 0.5]), initial_theta=np.array([1, 1]))[1])
    return sum(mle_accuracies) / len(mle_accuracies)


def scorematching_average(n, runs):
    """
    Calculates the average score matching accuracy over a number of runs
    n: sample size per run
    runs: number of runs
    """
    scorematching_accuracies = []
    for i in range(runs):
        print("method: scorematching_average", "n:", n, "run:", i)
        scorematching_accuracies.append(scorematching_accuracy(n, theta_star=np.array([1.5, 0.5]))[1])
    return sum(scorematching_accuracies) / len(scorematching_accuracies)


def generate_data_log_spacing(n_start, n_stop, num_ns, runs):
    ns = np.exp(np.linspace(math.log(n_start), math.log(n_stop), num=num_ns))
    ns = ns.astype(int)
    mle_accuracies = np.array([mle_average(n, runs) for n in ns])
    scorematch_accuracies = np.array([scorematching_average(n, runs) for n in ns])
    return (ns, mle_accuracies, scorematch_accuracies)


def write_to_file(ns, mle_accuracies, scorematch_accuracies, name='./results/log_data.txt'):
    f = open(name, 'w')
    for i in range(len(ns)):
        f.write(str(ns[i]) + ' ' + str(mle_accuracies[i]) + ' ' + str(scorematch_accuracies[i]) + '\n')
    f.close()


ns, mle_accuracies, scorematch_accuracies = generate_data_log_spacing(n_start=1000,
                                                                      n_stop=100000,
                                                                      num_ns=10,
                                                                      runs=100)
write_to_file(ns, mle_accuracies, scorematch_accuracies)
