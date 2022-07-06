import numpy as np
import math
import test_class
from run_tests import run_tests
from file_read_write import write_to_file
import sufficient_statistics


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


parameters_for_tests = asymptotic_test_parameters()
tests = run_tests(parameters_for_tests)
write_to_file(tests)
