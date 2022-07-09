import numpy as np
import math
import test_class
from run_tests import run_tests
from file_read_write import write_to_file, read_from_file
import sufficient_statistics
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.colors as colors
import matplotlib.cm as cm


#### DEFINE Experiment CLASS ####


class Experiment:
    def __init__(self, create_function, graph_function, description):
        self.create_function = create_function
        self.graph_function = graph_function
        self.description = description


experiments = []


#### CREATE HELPER FUNCTION ####


def graph_ellipse(axes, center, cov, color='black', nstd=1):
    eigvals, eigvecs = np.linalg.eigh(cov)
    orient = np.arctan2(eigvecs[:, 0][1], eigvecs[:, 0][0])
    ell = Ellipse(xy=center,
                    width=2 * nstd * np.sqrt(eigvals[0]),
                    height=2 * nstd * np.sqrt(eigvals[1]),
                    angle=np.degrees(orient), color=color)
    axes.add_artist(ell)
    ell.set_facecolor('none')
    return ell


#### CREATE EXPERIMENTS ####

#### CHANGING EXPONENT EXPERIMENT (NO LIMIT FORMULAS) ####


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


def accuracy_vs_exponent(tests):
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


def ellipses_vs_exponent(tests):
    #get limits from user:
    xmin = float(input("Enter xmin: "))
    xmax = float(input("Enter xmax: "))
    ymin = float(input("Enter ymin: "))
    ymax = float(input("Enter ymax: "))

    #create two sets of axes
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title('MLE')
    axes[1].set_title('Score Matching')
    for axis in axes:
        axis.set_xlim([xmin, xmax])
        axis.set_ylim([ymin, ymax])
        axis.set_xlabel('theta_0')
        axis.set_ylabel('theta_1')

    #calibrate the color scheme
    min_exponent = min([test.parameters.suffstats[1].exponent for test in tests])
    max_exponent = max([test.parameters.suffstats[1].exponent for test in tests])
    #add colorbar
    cmap = plt.get_cmap('jet')
    norm = colors.Normalize(vmin=min_exponent, vmax=max_exponent)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    mapper.set_array([])
    fig.colorbar(mapper, ax=axes[0], label='exponent')
    fig.colorbar(mapper, ax=axes[1], label='exponent')

    #plot the data
    for test in tests:
        if test.parameters.method == "mle":
            axis = axes[0]
        elif test.parameters.method == "scorematching":
            axis = axes[1]
        else:
            raise ValueError("Method not recognized")
        color = mapper.to_rgba(test.parameters.suffstats[1].exponent)
        ell = graph_ellipse(axis, test.results.mean + test.parameters.theta_star, test.results.cov, color=color)
        axis.plot([test.parameters.theta_star[0]], [test.parameters.theta_star[1]], '.', color='black')
    plt.show()


def graph_exponent_change(tests):
    action = input("Do you want to graph accuracy or ellipses? (a/e): ")
    if action == 'a':
        accuracy_vs_exponent(tests)
    elif action == 'e':
        ellipses_vs_exponent(tests)
    else:
        raise ValueError("Action not recognized")


changing_power_experiment_no_limits = Experiment(changing_exponent_parameters,
                                                    graph_exponent_change,
                                                    "Change the exponent of the polynomial (only mle and scorematching - no limits)")
experiments.append(changing_power_experiment_no_limits)


#### CHANGING n EXPERIMENT (NO LIMIT FORMULAS) ####


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


def accuracy_vs_n(tests):
    #tests is a list of test_class.Test objects
    log_mle_accuracies = []
    log_mle_ns = []
    log_scorematch_accuracies = []
    log_scorematch_ns = []

    for test in tests:
        if test.parameters.method == "mle":
            log_mle_accuracies.append(np.log(test.results.accuracy))
            log_mle_ns.append(np.log(test.parameters.n))
        elif test.parameters.method == "scorematching":
            log_scorematch_accuracies.append(np.log(test.results.accuracy))
            log_scorematch_ns.append(np.log(test.parameters.n))
        else:
            raise ValueError("Method not recognized")

    log_mle_accuracies = np.array(log_mle_accuracies)
    log_mle_ns = np.array(log_mle_ns)
    log_scorematch_accuracies = np.array(log_scorematch_accuracies)
    log_scorematch_ns = np.array(log_scorematch_ns)

    #create figure with two axes:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    raw = ax[0]
    adjusted = ax[1]

    #create subplot for plotting raw data:
    raw.plot(log_mle_ns, log_mle_accuracies, 'bo', label='MLE Data')
    raw.plot(log_mle_ns, log_mle_accuracies, 'b')
    raw.plot(log_scorematch_ns, log_scorematch_accuracies, 'ro', label='Score Matching Data')
    raw.plot(log_scorematch_ns, log_scorematch_accuracies, 'r')
    raw.set_xlabel('log(sample size)')
    raw.set_ylabel('log(accuracy)')
    raw.legend(loc='upper right')
    raw.set_title('Accuracy vs. Sample Size')

    #create subplot for plotting adjusted data:
    adjusted_mle_accuracies = log_mle_accuracies + log_mle_ns / 2
    adjusted_scorematch_accuracies = log_scorematch_accuracies + log_scorematch_ns / 2
    adjusted.plot(log_mle_ns, adjusted_mle_accuracies, 'bo', label='MLE Data')
    adjusted.plot(log_mle_ns, adjusted_mle_accuracies, 'b')
    adjusted.plot(log_scorematch_ns, adjusted_scorematch_accuracies, 'ro', label='Score Matching Data')
    adjusted.plot(log_scorematch_ns, adjusted_scorematch_accuracies, 'r')
    adjusted.set_xlabel('log(sample size)')
    adjusted.set_ylabel('log(accuracy) + log(sample size) / 2')
    adjusted.legend(loc='upper right')
    adjusted.set_title('Accuracy (adjusted) vs. Sample Size')

    #show the plot
    plt.show()


changing_n_experiment_no_limits = Experiment(asymptotic_test_parameters,
                                                accuracy_vs_n,
                                                "Change the sample size / test asymptotic behavior (only mle and scorematching - no limits)")
experiments.append(changing_n_experiment_no_limits)


#### CHANGING EXPONENT EXPERIMENT (mle and mle_limit) ####


def changing_exponent_mle_and_mle_limit_parameters():
    #collect inputs from user:
    n = int(input("Enter n: "))
    runs = int(input("Enter runs: "))
    exponent_start = int(input("Enter exponent start: "))
    exponent_stop = int(input("Enter exponent stop: "))
    num_exponents = int(input("Enter number of exponents: "))
    
    #generate parameters:
    exponents = np.exp(np.linspace(math.log(exponent_start), math.log(exponent_stop), num=num_exponents))
    exponents = 2 * (exponents / 2).astype(int)
    exponents = np.maximum(exponents, 4)
    print("Exponents: ", exponents)

    parameters_for_tests = []
    for method in ['mle', 'mle_limit']:
        for exponent in exponents:        
            suffstats = [sufficient_statistics.FirstStat(), sufficient_statistics.PolyStat(exponent)]
            parameters_for_tests.append(test_class.TestParameters(suffstats, np.array([1.0, 1.0]), n, method, runs))
    return parameters_for_tests


def ellipses_vs_exponent_mle_and_mle_limit(tests):
    #get limits from user:
    xmin = float(input("Enter xmin: "))
    xmax = float(input("Enter xmax: "))
    ymin = float(input("Enter ymin: "))
    ymax = float(input("Enter ymax: "))

    #create two sets of axes
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title('MLE')
    axes[1].set_title('MLE (limit formula)')
    for axis in axes:
        axis.set_xlim([xmin, xmax])
        axis.set_ylim([ymin, ymax])
        axis.set_xlabel('theta_0')
        axis.set_ylabel('theta_1')

    #calibrate the color scheme
    min_exponent = min([test.parameters.suffstats[1].exponent for test in tests])
    max_exponent = max([test.parameters.suffstats[1].exponent for test in tests])
    #add colorbar
    cmap = plt.get_cmap('jet')
    norm = colors.Normalize(vmin=min_exponent, vmax=max_exponent)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    mapper.set_array([])
    fig.colorbar(mapper, ax=axes[0], label='exponent')
    fig.colorbar(mapper, ax=axes[1], label='exponent')

    #plot the data
    for test in tests:
        if test.parameters.method == "mle":
            axis = axes[0]
        elif test.parameters.method == "mle_limit":
            axis = axes[1]
        else:
            raise ValueError("Method not recognized")
        color = mapper.to_rgba(test.parameters.suffstats[1].exponent)
        ell = graph_ellipse(axis, test.results.mean + test.parameters.theta_star, test.results.cov, color=color)
        axis.plot([test.parameters.theta_star[0]], [test.parameters.theta_star[1]], '.', color='black')
    plt.show()


changing_power_experiment_mle_and_mle_limit = Experiment(changing_exponent_mle_and_mle_limit_parameters,
                                                            ellipses_vs_exponent_mle_and_mle_limit,
                                                            "Change the exponent of the polynomial (between mle and mle_limit -- no scorematching)")
experiments.append(changing_power_experiment_mle_and_mle_limit)


#### CHANGING EXPONENT EXPERIMENT (scorematching and scorematching_limit) ####


def changing_exponent_scorematching_and_scorematching_limit_parameters():
    #collect inputs from user:
    n = int(input("Enter n: "))
    runs = int(input("Enter runs: "))
    exponent_start = int(input("Enter exponent start: "))
    exponent_stop = int(input("Enter exponent stop: "))
    num_exponents = int(input("Enter number of exponents: "))
    
    #generate parameters:
    exponents = np.exp(np.linspace(math.log(exponent_start), math.log(exponent_stop), num=num_exponents))
    exponents = 2 * (exponents / 2).astype(int)
    exponents = np.maximum(exponents, 4)
    print("Exponents: ", exponents)

    parameters_for_tests = []
    for method in ['scorematching_limit', 'scorematching']:
        for exponent in exponents:        
            suffstats = [sufficient_statistics.FirstStat(), sufficient_statistics.PolyStat(exponent)]
            parameters_for_tests.append(test_class.TestParameters(suffstats, np.array([1.0, 1.0]), n, method, runs))
    return parameters_for_tests


def ellipses_vs_exponent_scorematching_and_scorematching_limit(tests):
    #get limits from user:
    xmin = float(input("Enter xmin: "))
    xmax = float(input("Enter xmax: "))
    ymin = float(input("Enter ymin: "))
    ymax = float(input("Enter ymax: "))

    #create two sets of axes
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title('Score Matching')
    axes[1].set_title('Score Matching (limit formula)')
    for axis in axes:
        axis.set_xlim([xmin, xmax])
        axis.set_ylim([ymin, ymax])
        axis.set_xlabel('theta_0')
        axis.set_ylabel('theta_1')

    #calibrate the color scheme
    min_exponent = min([test.parameters.suffstats[1].exponent for test in tests])
    max_exponent = max([test.parameters.suffstats[1].exponent for test in tests])
    #add colorbar
    cmap = plt.get_cmap('jet')
    norm = colors.Normalize(vmin=min_exponent, vmax=max_exponent)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    mapper.set_array([])
    fig.colorbar(mapper, ax=axes[0], label='exponent')
    fig.colorbar(mapper, ax=axes[1], label='exponent')

    #plot the data
    for test in tests:
        if test.parameters.method == "scorematching":
            axis = axes[0]
        elif test.parameters.method == "scorematching_limit":
            axis = axes[1]
        else:
            raise ValueError("Method not recognized")
        color = mapper.to_rgba(test.parameters.suffstats[1].exponent)
        ell = graph_ellipse(axis, test.results.mean + test.parameters.theta_star, test.results.cov, color=color)
        axis.plot([test.parameters.theta_star[0]], [test.parameters.theta_star[1]], '.', color='black')
    plt.show()


changing_power_experiment_scorematching_and_scorematching_limit = Experiment(changing_exponent_scorematching_and_scorematching_limit_parameters,
                                                                                ellipses_vs_exponent_scorematching_and_scorematching_limit,
                                                                                "Change the exponent of the polynomial (between scorematching and scorematching_limit -- no mle)")
experiments.append(changing_power_experiment_scorematching_and_scorematching_limit)



#### CHANGING EXPONENT EXPERIMENT (mle_limit and scorematching_limit) ####


def changing_exponent_mle_limit_and_scorematching_limit_parameters():
    #collect inputs from user:
    n = int(input("Enter n: "))
    runs = int(input("Enter runs: "))
    exponent_start = int(input("Enter exponent start: "))
    exponent_stop = int(input("Enter exponent stop: "))
    num_exponents = int(input("Enter number of exponents: "))
    
    #generate parameters:
    exponents = np.exp(np.linspace(math.log(exponent_start), math.log(exponent_stop), num=num_exponents))
    exponents = 2 * (exponents / 2).astype(int)
    exponents = np.maximum(exponents, 4)
    print("Exponents: ", exponents)

    parameters_for_tests = []
    for method in ['mle_limit', 'scorematching_limit']:
        for exponent in exponents:        
            suffstats = [sufficient_statistics.FirstStat(), sufficient_statistics.PolyStat(exponent)]
            parameters_for_tests.append(test_class.TestParameters(suffstats, np.array([1.0, 1.0]), n, method, runs))
    return parameters_for_tests


def ellipses_vs_exponent_mle_limit_and_scorematching_limit(tests):
    #get limits from user:
    xmin = float(input("Enter xmin: "))
    xmax = float(input("Enter xmax: "))
    ymin = float(input("Enter ymin: "))
    ymax = float(input("Enter ymax: "))

    #create two sets of axes
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title('MLE (limit formula)')
    axes[1].set_title('Score Matching (limit formula)')
    for axis in axes:
        axis.set_xlim([xmin, xmax])
        axis.set_ylim([ymin, ymax])
        axis.set_xlabel('theta_0')
        axis.set_ylabel('theta_1')

    #calibrate the color scheme
    min_exponent = min([test.parameters.suffstats[1].exponent for test in tests])
    max_exponent = max([test.parameters.suffstats[1].exponent for test in tests])
    #add colorbar
    cmap = plt.get_cmap('jet')
    norm = colors.Normalize(vmin=min_exponent, vmax=max_exponent)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    mapper.set_array([])
    fig.colorbar(mapper, ax=axes[0], label='exponent')
    fig.colorbar(mapper, ax=axes[1], label='exponent')

    #plot the data
    for test in tests:
        print("method: ", test.parameters.method,
                "exponent: ", test.parameters.suffstats[1].exponent,
                "covariance: ", test.results.cov)
        if test.parameters.method == "mle_limit":
            axis = axes[0]
        elif test.parameters.method == "scorematching_limit":
            axis = axes[1]
        else:
            raise ValueError("Method not recognized")
        color = mapper.to_rgba(test.parameters.suffstats[1].exponent)
        ell = graph_ellipse(axis, test.results.mean + test.parameters.theta_star, test.results.cov, color=color)
        axis.plot([test.parameters.theta_star[0]], [test.parameters.theta_star[1]], '.', color='black')
    plt.show()


def accuracy_vs_exponent_mle_limit_and_scorematching_limit(tests):
    #tests is a list of test_class.Test objects
    log_mle_limit_accuracies = []
    log_mle_limit_exponents = []
    log_scorematch_limit_accuracies = []
    log_scorematch_limit_exponents = []
    for test in tests:
        if test.parameters.method == "mle_limit":
            log_mle_limit_accuracies.append(np.log(test.results.accuracy))
            log_mle_limit_exponents.append(np.log(test.parameters.suffstats[1].exponent))
        elif test.parameters.method == "scorematching_limit":
            log_scorematch_limit_accuracies.append(np.log(test.results.accuracy))
            log_scorematch_limit_exponents.append(np.log(test.parameters.suffstats[1].exponent))
        else:
            raise ValueError("Method not recognized")
    #plot the data:
    plt.plot(log_mle_limit_exponents, log_mle_limit_accuracies, 'bo', label='MLE Limit Data')
    plt.plot(log_scorematch_limit_exponents, log_scorematch_limit_accuracies, 'ro', label='Score Matching Limit Data')
    #label the axes:
    plt.xlabel('log(exponent)')
    plt.ylabel('log(accuracy)')
    plt.legend(loc='upper left')
    plt.show()


def graph_exponent_change_mle_limit_and_scorematching_limit(tests):
    action = input("Do you want to graph accuracy or ellipses? (a/e): ")
    if action == 'a':
        accuracy_vs_exponent_mle_limit_and_scorematching_limit(tests)
    elif action == 'e':
        ellipses_vs_exponent_mle_limit_and_scorematching_limit(tests)
    else:
        raise ValueError("Action not recognized")



changing_power_experiment_mle_limit_and_scorematching_limit = Experiment(changing_exponent_mle_limit_and_scorematching_limit_parameters,
                                                                            graph_exponent_change_mle_limit_and_scorematching_limit,
                                                                            "Change the exponent of the polynomial (between mle_limit and scorematching_limit)")
experiments.append(changing_power_experiment_mle_limit_and_scorematching_limit)


#### SINUSOID EXPERIMENT (mle_limit and scorematching_limit) ####


def sinusoid_mle_limit_and_scorematching_limit_parameters():
    #collect inputs from user:
    n = int(input("Enter n: "))
    runs = int(input("Enter runs: "))
    exponent_start = float(input("Enter frequency start: "))
    exponent_stop = float(input("Enter frequency stop: "))
    num_exponents = int(input("Enter number of frequencies: "))
    
    #generate parameters:
    freqiencies = np.exp(np.linspace(math.log(exponent_start), math.log(exponent_stop), num=num_exponents))

    parameters_for_tests = []
    for method in ['mle_limit', 'scorematching_limit']:
        for frequency in freqiencies:
            suffstats = [sufficient_statistics.FirstStat(), sufficient_statistics.SinusoidStat(amplitude=1, frequency=frequency, phase=0)]
            parameters_for_tests.append(test_class.TestParameters(suffstats, np.array([1.0, 1.0]), n, method, runs))
    return parameters_for_tests


def ellipses_vs_frequency_mle_limit_and_scorematching_limit(tests):
    #get limits from user:
    xmin = float(input("Enter xmin: "))
    xmax = float(input("Enter xmax: "))
    ymin = float(input("Enter ymin: "))
    ymax = float(input("Enter ymax: "))

    #create two sets of axes
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title('MLE (limit formula)')
    axes[1].set_title('Score Matching (limit formula)')
    for axis in axes:
        axis.set_xlim([xmin, xmax])
        axis.set_ylim([ymin, ymax])
        axis.set_xlabel('theta_0')
        axis.set_ylabel('theta_1')

    #calibrate the color scheme
    min_frequency = min([test.parameters.suffstats[1].frequency for test in tests])
    max_frequency = max([test.parameters.suffstats[1].frequency for test in tests])
    #add colorbar
    cmap = plt.get_cmap('jet')
    norm = colors.Normalize(vmin=min_frequency, vmax=max_frequency)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    mapper.set_array([])
    fig.colorbar(mapper, ax=axes[0], label='frequency')
    fig.colorbar(mapper, ax=axes[1], label='frequency')

    #plot the data
    for test in tests:
        print("method: ", test.parameters.method,
                "frequency: ", test.parameters.suffstats[1].frequency,
                "covariance: ", test.results.cov)
        if test.parameters.method == "mle_limit":
            axis = axes[0]
        elif test.parameters.method == "scorematching_limit":
            axis = axes[1]
        else:
            raise ValueError("Method not recognized")
        color = mapper.to_rgba(test.parameters.suffstats[1].frequency)
        ell = graph_ellipse(axis, test.results.mean + test.parameters.theta_star, test.results.cov, color=color)
        axis.plot([test.parameters.theta_star[0]], [test.parameters.theta_star[1]], '.', color='black')
    plt.show()


def accuracy_vs_frequency_mle_limit_and_scorematching_limit(tests):
    #tests is a list of test_class.Test objects
    log_mle_limit_accuracies = []
    log_mle_limit_frequencies = []
    log_scorematch_limit_accuracies = []
    log_scorematch_limit_frequencies = []
    for test in tests:
        if test.parameters.method == "mle_limit":
            log_mle_limit_accuracies.append(np.log(test.results.accuracy))
            log_mle_limit_frequencies.append(np.log(test.parameters.suffstats[1].frequency))
        elif test.parameters.method == "scorematching_limit":
            log_scorematch_limit_accuracies.append(np.log(test.results.accuracy))
            log_scorematch_limit_frequencies.append(np.log(test.parameters.suffstats[1].frequency))
        else:
            raise ValueError("Method not recognized")
    #plot the data:
    plt.plot(log_mle_limit_frequencies, log_mle_limit_accuracies, 'bo', label='MLE Limit Data')
    plt.plot(log_scorematch_limit_frequencies, log_scorematch_limit_accuracies, 'ro', label='Score Matching Limit Data')
    #label the axes:
    plt.xlabel('log(frequency)')
    plt.ylabel('log(accuracy)')
    plt.legend(loc='upper left')
    plt.show()


def graph_frequency_change_mle_limit_and_scorematching_limit(tests):
    action = input("Do you want to graph accuracy or ellipses? (a/e): ")
    if action == 'a':
        accuracy_vs_frequency_mle_limit_and_scorematching_limit(tests)
    elif action == 'e':
        ellipses_vs_frequency_mle_limit_and_scorematching_limit(tests)
    else:
        raise ValueError("Action not recognized")



changing_frequency_experiment_mle_limit_and_scorematching_limit = Experiment(sinusoid_mle_limit_and_scorematching_limit_parameters,
                                                                                graph_frequency_change_mle_limit_and_scorematching_limit,
                                                                                "Change the frequency of a sinusoid (between mle_limit and scorematching_limit)")
experiments.append(changing_frequency_experiment_mle_limit_and_scorematching_limit)


#### USER INTERACTION ####


def create_query_user():
    query_string = "What experiment would you like to run?\n"
    for i, experiment in enumerate(experiments):
        query_string += str(i) + ": " + experiment.description + "\n"
    query_string += "Enter the kind of experiment you would like to run (integer): "
    action = int(input(query_string))
    parameters_for_tests = experiments[action].create_function()
    tests = run_tests(parameters_for_tests)
    write_to_file(tests)


def graph_query_user():
    test_number = input("Enter test number: ")
    tests = read_from_file(test_number)
    query_string = "What kinda of experiment was this?\n"
    for i, experiment in enumerate(experiments):
        query_string += str(i) + ": " + experiment.description + "\n"
    query_string += "Enter an integer: "
    action = int(input(query_string))
    experiments[action].graph_function(tests)


def query_user():
    create_or_graph = input("Do you want to create a new test or graph an existing test? (c/g): ")
    if create_or_graph == "c":
        create_query_user()
    elif create_or_graph == "g":
        graph_query_user()
    else:
        raise ValueError("Action not recognized")


query_user()
