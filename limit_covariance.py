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
            def suffstats_product(x):
                return row_suffstat.zeroth_derivative(x) * column_suffstat.zeroth_derivative(x)
            entry = expectation(suffstats_product, test_parameters, Z)
            E_FxFxT_row.append(entry)
        E_FxFxT.append(E_FxFxT_row)
    E_FxFxT = np.array(E_FxFxT)

    #Now, let's compute the Fisher matrix.
    Fisher = E_FxFxT - E_Fx_E_FxT

    #Finally, we return the inverse of the Fisher matrix
    #after scaling for sample size.
    return np.linalg.inv(Fisher) / test_parameters.n


def scorematching_limit_covariance(test_parameters):
    #test_parameters is of type test_class.TestParameters.

    #First, compute the partition function.
    Z = partition(test_parameters)

    #Let's now compute the "outer" matrix
    E_JFxJFxT = []
    for row_suffstat in test_parameters.suffstats:
        E_JFxJFxT_row = []
        for column_suffstat in test_parameters.suffstats:
            def first_derivatives_product(x):
                return (-row_suffstat.first_derivative(x)) * (-column_suffstat.first_derivative(x))
            entry = expectation(first_derivatives_product, test_parameters, Z)
            E_JFxJFxT_row.append(entry)
        E_JFxJFxT.append(E_JFxJFxT_row)
    E_JFxJFxT = np.array(E_JFxJFxT)
    outer_matrix = np.linalg.inv(E_JFxJFxT)

    #Now, let's compute the "inner" matrix
    #We'll start by defining some of the functions involved.
    def A(x):
        JFx = -ss.first_derivatives(test_parameters.suffstats, np.array([x]))
        JFx_JFxT = np.outer(JFx, JFx)
        JFx_JFxT_theta = np.dot(JFx_JFxT, -test_parameters.theta_star)
        Delta_F = -ss.second_derivatives(test_parameters.suffstats, np.array([x]))
        return JFx_JFxT_theta - Delta_F

    #This is computationally stupid because we compute all of A
    #each time we ask for each component, but it's a bit nicer to think about.
    As = [lambda x: A(x)[i] for i in range(len(test_parameters.suffstats))]

    #We can now duplicate the code from the MLE case to get E[A(x) A(x)^T] - E[A(x)] E[A(x)^T]
    #i.e. Sigma_A

    E_Ax = [expectation(As_i, test_parameters, Z) for As_i in As]
    E_Ax = np.array(E_Ax)
    E_Ax_E_AxT = np.outer(E_Ax, E_Ax)

    E_AxAxT = []
    for As_i in As:
        E_AxAxT_row = []
        for As_j in As:
            def As_i_As_j_product(x):
                return As_i(x) * As_j(x)
            entry = expectation(As_i_As_j_product, test_parameters, Z)
            E_AxAxT_row.append(entry)
        E_AxAxT.append(E_AxAxT_row)
    E_AxAxT = np.array(E_AxAxT)

    Sigma_A = E_AxAxT - E_Ax_E_AxT

    #Finally we can compute our answer:
    #(Note: this is actually computing the covariance matrix for -theta,
    #but that's the same as the covariance matrix for theta.)
    return np.matmul(outer_matrix, np.matmul(Sigma_A, outer_matrix)) / test_parameters.n
