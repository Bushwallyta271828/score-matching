import matplotlib.pyplot as plt
import numpy as np


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
    #plot the best fit lines
    plt.plot(log_ns, mle_slope*log_ns + mle_intercept, 'b', label='MLE best fit')
    plt.plot(log_ns, scorematch_slope*log_ns + scorematch_intercept, 'r', label='Scorematch best fit')
    #plot the original data:
    plt.plot(log_ns, log_mles, 'bo', label='MLE data')
    plt.plot(log_ns, log_scorematches, 'ro', label='Scorematch data')
    #label the axes:
    plt.xlabel('log(sample size)')
    plt.ylabel('log(distance in parameter space)')
    #create a legend with correct colors:
    plt.legend(loc='upper right')
    #show the plot
    plt.show()

    #print the slope and intercept
    print("MLE slope:", mle_slope)
    print("MLE intercept:", mle_intercept)
    print("Score Matching slope:", scorematch_slope)
    print("Score Matching intercept:", scorematch_intercept)
    return (mle_slope, mle_intercept, scorematch_slope, scorematch_intercept)


ns, mle_accuracies, scorematch_accuracies = read_from_file()
best_fit(ns, mle_accuracies, scorematch_accuracies)
