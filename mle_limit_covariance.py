import numpy as np
import sufficient_statistics as ss
import test_class
from scipy.integrate import quad


def non_normalized(x, test_parameters):
    exponent = np.dot(test_parameters.theta_star, ss.zeroth_derivatives(test_parameters.suffstats, np.array([x])))[0]
    return np.exp(exponent)


def partition(test_parameters):
    return quad(non_normalized, -np.inf, np.inf, args=(test_parameters))[0]


def normalized(x, test_parameters, Z):
    return non_normalized(x, test_parameters) / Z


def weighted_function(x, function, test_parameters, Z):
    #assumes function wants a numpy array as argument
    return normalized(x, test_parameters, Z) * function(np.array([x]))[0]


def expectation(function, test_parameters, Z):
    return quad(weighted_function, -np.inf, np.inf, args=(function, test_parameters, Z))[0]




#def expectation_function(x, suffstat_index, test_parameters):
#    #COME BACK THIS IS WRONG!!!
#    #This is the function that is integrated to compute E[F(x)]
#    #x is the point at which the function is evaluated
#    #suffstat_index is the index of the suffstat that is being integrated over
#    #test_parameters is of type test_class.TestParameters.
#    #The function returns the value of the function at x.
#    return test_parameters.F(x) * test_parameters.suffstats[suffstat_index].get_coefficient(x)


def mle_limit_covariance(test_parameters):
    #test_parameters is of type test_class.TestParameters.
    #The formula for the Fisher matrix is E[F(x) F(x)^T] - E[F(x)] E[F(x)^T].
    #The covariance should then be the inverse of the Fisher matrix (scaled for sample size).
    #Note that because F appears twice in each term, it is okay to use
    #the negative of F instead. This is important because my code currently
    #uses the convention that sufficient statistics are non-positive
    #and the coefficients are non-negative, as opposed to the LaTeX document
    #which has the opposite convention.

    #First, compute the partition function.
    Z = partition(test_parameters)

    #Now, let's compute E[F(x)] and thus E[F(x)] E[F(x)^T].
    E_Fx = [expectation(suffstat.zeroth_derivative, test_parameters, Z) for suffstat in test_parameters.suffstats]
    E_Fx = np.array(E_Fx)
    E_Fx_E_FxT = np.outer(E_Fx, E_Fx)

    #Now, let's compute E[F(x) F(x)^T]
    E_FxFxT = []
    for row_suffstat in test_parameters.suffstats:
        E_FxFxT_row = []
        for column_suffstat in test_parameters.suffstats:
            def product_suffstat(x):
                return row_suffstat.zeroth_derivative(x) * column_suffstat.zeroth_derivative(x)
            entry = expectation(product_suffstat, test_parameters, Z)
            E_FxFxT_row.append(entry)
        E_FxFxT.append(E_FxFxT_row)
    E_FxFxT = np.array(E_FxFxT)

    #Now, let's compute the Fisher matrix.
    Fisher = E_FxFxT - E_Fx_E_FxT

    #Finally, we return the inverse of the Fisher matrix
    #after scaling for sample size.
    return np.linalg.inv(Fisher) / test_parameters.n


def scorematching_limit_covariance(test_parameters):
    #COME BACK
