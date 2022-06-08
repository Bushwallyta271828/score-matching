from mle_accuracy import mle_accuracy
from scorematching_accuracy import scorematching_accuracy
import matplotlib.pyplot as plt
import numpy as np
import math


def mle_average(n, runs):
    """
    Calculates the average MLE accuracy over a number of runs
    :param n: number of trials
    :param runs: number of runs
    :return: average MLE accuracy
    """
    mle_accuracies = []
    for i in range(runs):
        print("Run:", i, "n:", n, "method: mle_average")
        mle_accuracies.append(mle_accuracy(n, [0.15, 0.05])[1])
    return sum(mle_accuracies) / len(mle_accuracies)


def scorematching_average(n, runs):
    """
    Calculates the average score matching accuracy over a number of runs
    :param n: number of trials
    :param runs: number of runs
    :return: average score matching accuracy
    """
    scorematching_accuracies = []
    for i in range(runs):
        print("Run:", i, "n:", n, "method: scorematching_average")
        scorematching_accuracies.append(scorematching_accuracy(n, [0.15, 0.05])[1])
    return sum(scorematching_accuracies) / len(scorematching_accuracies)


def generate_data_log_spacing(n_start, n_stop, num_ns, runs):
    ns = np.exp(np.linspace(math.log(n_start), math.log(n_stop), num=num_ns))
    ns = ns.astype(int)
    print("ns:", ns)
    #vectorize the functions
    mle_accuracies = np.vectorize(mle_average)
    scorematch_accuracies = np.vectorize(scorematching_average)
    #calculate the accuracies
    mle_accuracies = mle_accuracies(ns, runs)
    scorematch_accuracies = scorematch_accuracies(ns, runs)
    return (ns, mle_accuracies, scorematch_accuracies)


def write_to_file(n_start, n_stop, datapoints, runs):
    f = open('log_data.txt', 'w')
    output = generate_log_data(n_start, n_stop, datapoints, runs)
    ns = output[0]
    mle_accuracies = output[1]
    scorematch_accuracies = output[2]
    for i in range(len(ns)):
        f.write(str(ns[i]) + ' ' + str(mle_accuracies[i]) + ' ' + str(scorematch_accuracies[i]) + '\n')
    f.close()


def read_from_file():
    """
    Reads the data from the file log_data.txt
    """
    f = open('log_data.txt', 'r')
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
    plt.legend(['MLE', 'MLE best fit', 'Score Matching', 'Score Matching best fit'], loc='upper left')
    #show the plot
    plt.show()

    #print the slope and intercept
    print("MLE slope:", mle_slope)
    print("MLE intercept:", mle_intercept)
    print("Score Matching slope:", scorematch_slope)
    print("Score Matching intercept:", scorematch_intercept)
    return (mle_slope, mle_intercept, scorematch_slope, scorematch_intercept)



#print(generate_data_log_spacing(10, 100, 10, 10))
ns, mle_accuracies, scorematch_accuracies = generate_data_log_spacing(n_start=10,
                                                                      n_stop=100,
                                                                      num_ns=10,
                                                                      runs=100)
best_fit(ns, mle_accuracies, scorematch_accuracies)
