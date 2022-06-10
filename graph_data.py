import matplotlib.pyplot as plt
import numpy as np
from file_read_write import read_from_file


#Old code:
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

def graph_changing_theta1(methods, ns, runs, theta_stars, accuracies, means, covs):
    mle_accuracies = []
    mle_theta_1s = []
    scorematch_accuracies = []
    scorematch_theta_1s = []
    for i in range(len(theta_stars)):
        if methods[i] == "mle":
            mle_accuracies.append(np.log(accuracies[i]))
            mle_theta_1s.append(theta_stars[i][1])
        elif methods[i] == "scorematching":
            scorematch_accuracies.append(np.log(accuracies[i]))
            scorematch_theta_1s.append(theta_stars[i][1])
        else:
            raise ValueError("Method not recognized")
    #plot the data:
    plt.plot(mle_theta_1s, mle_accuracies, 'bo', label='MLE data')
    plt.plot(scorematch_theta_1s, scorematch_accuracies, 'ro', label='Scorematch data')
    #label the axes:
    plt.xlabel('theta_1')
    plt.ylabel('log(accuracy)')
    #create a legend with correct colors:
    plt.legend(loc='upper left')
    #show the plot
    plt.show()


test_number = input("Enter test number: ")
methods, ns, runs, theta_stars, accuracies, means, covs = read_from_file(test_number)
graph_changing_theta1(methods, ns, runs, theta_stars, accuracies, means, covs)
