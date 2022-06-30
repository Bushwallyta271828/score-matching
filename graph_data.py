import matplotlib.pyplot as plt
import numpy as np
from file_read_write import read_from_file
from matplotlib.patches import Ellipse
import test_class
import sufficient_statistics



#Old code:
#def best_fit(ns, mle_accuracies, scorematch_accuracies):
#    """
#    Finds the best fit line for the output from log_log_plot
#    for both mle_average and scorematching_average
#    """
#    log_ns = np.log(ns)
#    log_mles = np.log(mle_accuracies)
#    log_scorematches = np.log(scorematch_accuracies)
#    #calculate slope and intercept
#    mle_slope, mle_intercept = np.polyfit(log_ns, log_mles, 1)
#    scorematch_slope, scorematch_intercept = np.polyfit(log_ns, log_scorematches, 1)
#    #plot the best fit lines
#    plt.plot(log_ns, mle_slope*log_ns + mle_intercept, 'b', label='MLE best fit')
#    plt.plot(log_ns, scorematch_slope*log_ns + scorematch_intercept, 'r', label='Scorematch best fit')
#    #plot the original data:
#    plt.plot(log_ns, log_mles, 'bo', label='MLE data')
#    plt.plot(log_ns, log_scorematches, 'ro', label='Scorematch data')
#    #label the axes:
#    plt.xlabel('log(sample size)')
#    plt.ylabel('log(distance in parameter space)')
#    #create a legend with correct colors:
#    plt.legend(loc='upper right')
#    #show the plot
#    plt.show()
#
#    #print the slope and intercept
#    print("MLE slope:", mle_slope)
#    print("MLE intercept:", mle_intercept)
#    print("Score Matching slope:", scorematch_slope)
#    print("Score Matching intercept:", scorematch_intercept)
#    return (mle_slope, mle_intercept, scorematch_slope, scorematch_intercept)


#def graph_changing_theta1(methods, ns, runs, theta_stars, accuracies, means, covs):
#    log_mle_accuracies = []
#    log_mle_theta_1s = []
#    log_scorematch_accuracies = []
#    log_scorematch_theta_1s = []
#    for i in range(len(theta_stars)):
#        if methods[i] == "mle":
#            log_mle_accuracies.append(np.log(accuracies[i]))
#            log_mle_theta_1s.append(np.log(theta_stars[i][1]))
#        elif methods[i] == "scorematching":
#            log_scorematch_accuracies.append(np.log(accuracies[i]))
#            log_scorematch_theta_1s.append(np.log(theta_stars[i][1]))
#        else:
#            raise ValueError("Method not recognized")
#    #plot the data:
#    plt.plot(log_mle_theta_1s, log_mle_accuracies, 'bo', label='MLE Data')
#    plt.plot(log_scorematch_theta_1s, log_scorematch_accuracies, 'ro', label='Score Matching Data')
#    #label the axes:
#    plt.xlabel('log(theta_1)')
#    plt.ylabel('log(accuracy)')
#    plt.legend(loc='upper left')
#    plt.show()


#def graph_ellipses(methods, ns, runs, theta_stars, accuracies, means, covs, xmin, xmax, ymin, ymax):
#    ax = plt.gca()
#    drawn_mle = False
#    drawn_scorematch = False
#    for i in range(len(methods)):
#        eigvals, eigvecs = np.linalg.eigh(covs[i])
#        orient = np.arctan2(eigvecs[:, 0][1], eigvecs[:, 0][0])
#        nsdt = 1
#        if methods[i] == "mle":
#            color = 'b'
#        elif methods[i] == "scorematching":
#            color = 'r'
#        else:
#            raise ValueError("Method not recognized")
#        ell = Ellipse(xy=means[i] + theta_stars[i],
#                        width=2 * nsdt * np.sqrt(eigvals[0]),
#                        height=2 * nsdt * np.sqrt(eigvals[1]),
#                        angle=np.degrees(orient), color=color)
#        if methods[i] == "mle" and not drawn_mle:
#            ell.set_label('MLE')
#            drawn_mle = True
#        elif methods[i] == "scorematching" and not drawn_scorematch:
#            ell.set_label('Score Matching')
#            drawn_scorematch = True
#        ax.add_artist(ell)
#        ell.set_facecolor('none')
#
#        ax.plot([theta_stars[i][0]], [theta_stars[i][1]], 'o', color='black')
#
#    ax.set_xlim([xmin, xmax])
#    ax.set_ylim([ymin, ymax])
#    ax.set_xlabel('theta_0')
#    ax.set_ylabel('theta_1')
#    ax.legend(loc='upper right')
#    #show the plot
#    plt.show()


def graph_changing_exponent(tests):
    #tests is a list of test_class.Test objects
    log_mle_accuracies = []
    log_mle_exponents = []
    log_scorematch_accuracies = []
    log_scorematch_exponents = []
    for test in tests:
        if test.parameters.method == "mle":
            log_mle_accuracies.append(np.log(test.results.accuracy))
            log_mle_exponents.append(np.log(test.parameters.suffstats[1].exponent))
        elif test.parameters.method == "scorematching":
            log_scorematch_accuracies.append(np.log(test.results.accuracy))
            log_scorematch_exponents.append(np.log(test.parameters.suffstats[1].exponent))
        else:
            raise ValueError("Method not recognized")
    #plot the data:
    plt.plot(log_mle_exponents, log_mle_accuracies, 'bo', label='MLE Data')
    plt.plot(log_scorematch_exponents, log_scorematch_accuracies, 'ro', label='Score Matching Data')
    #label the axes:
    plt.xlabel('log(exponent)')
    plt.ylabel('log(accuracy)')
    plt.legend(loc='upper left')
    plt.show()


def query_user():
    test_number = input("Enter test number: ")
    tests = read_from_file(test_number)
    query_string = "What would you like to do?\n"
    query_string += "1. Fit changing n to asymptotic regime\n"
    query_string += "2. Graph accuracy vs. theta_1\n"
    query_string += "3. Graph accuracy vs. exponent\n"
    query_string += "4. Draw ellipses for changing theta_1\n"
    query_string += "5. Draw ellipses for changing exponent\n"
    action = input(query_string)
    if action == "1":
        accuracy_vs_n(tests)
    elif action == "2":
        accuracy_vs_theta1(tests)
    elif action == "3":
        accuracy_vs_exponent(tests)
    elif action == "4":
        ellipses_vs_theta1(tests)
    elif action == "5":
        ellipses_vs_exponent(tests)
    else:
        raise ValueError("Action not recognized")

    #    raise ValueError("Ellipse graph not yet implemented")
    #    #get xlim and ylim:
    #    xmin = float(input("Enter xmin: "))
    #    xmax = float(input("Enter xmax: "))
    #    ymin = float(input("Enter ymin: "))
    #    ymax = float(input("Enter ymax: "))
    #    graph_ellipses(methods, ns, runs, theta_stars, accuracies, means, covs, xmin, xmax, ymin, ymax)


query_user()
