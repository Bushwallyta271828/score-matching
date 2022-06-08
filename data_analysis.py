from mle_accuracy import mle_accuracy
from scorematching_accuracy import scorematching_accuracy
import matplotlib.pyplot as plt
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


def read_from_file(name='./results/log_data.txt'):
    f = open(name, 'r')
    ns = []
    mle_accuracies = []
    scorematch_accuracies = []
    for line in f:
        line = line.split()
        ns.append(int(line[0]))
        mle_accuracies.append(float(line[1]))
        scorematch_accuracies.append(float(line[2]))
    f.close()
    ns = np.array(ns, dtype=int)
    mle_accuracies = np.array(mle_accuracies, dtype=float)
    scorematch_accuracies = np.array(scorematch_accuracies, dtype=float)
    return (ns, mle_accuracies, scorematch_accuracies)


def best_fit(ns, mle_accuracies, scorematch_accuracies):
    """
    Finds the best fit line for the output from log_log_plot
    for both mle_average and scorematching_average
    """
    log_ns = np.log(ns)
    log_mles = np.log(mle_accuracies)
    log_scorematches = np.log(scorematch_accuracies)
    #calculate slope and intercept
    mle_slope, mle_intercept = np.polyfit(log_ns, log_mles, 1)
    scorematch_slope, scorematch_intercept = np.polyfit(log_ns, log_scorematches, 1)
    #plot the best fit line
    plt.plot(log_ns, log_mles, 'b')
    plt.plot(log_ns, mle_slope*log_ns + mle_intercept, 'b')
    plt.plot(log_ns, log_scorematches, 'r')
    plt.plot(log_ns, scorematch_slope*log_ns + scorematch_intercept, 'r')
    #plot the original data:
    plt.plot(log_ns, log_mles, 'bo')
    plt.plot(log_ns, log_scorematches, 'ro')
    #label the axes:
    plt.xlabel('log(sample size)')
    plt.ylabel('log(distance in parameter space)')
    #create a legend with correct colors:
    plt.legend(['MLE', 'MLE best fit', 'Score Matching', 'Score Matching best fit'], loc='upper right')
    #show the plot
    plt.show()

    #print the slope and intercept
    print("MLE slope:", mle_slope)
    print("MLE intercept:", mle_intercept)
    print("Score Matching slope:", scorematch_slope)
    print("Score Matching intercept:", scorematch_intercept)
    return (mle_slope, mle_intercept, scorematch_slope, scorematch_intercept)



#ns, mle_accuracies, scorematch_accuracies = generate_data_log_spacing(n_start=1000,
#                                                                      n_stop=100000,
#                                                                      num_ns=10,
#                                                                      runs=100)
#write_to_file(ns, mle_accuracies, scorematch_accuracies)
ns, mle_accuracies, scorematch_accuracies = read_from_file()
best_fit(ns, mle_accuracies, scorematch_accuracies)
