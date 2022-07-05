import numpy as np
import sufficient_statistics as ss
import test_class
from scipy.integrate import quad


#def expectation_function(x, suffstat_index, test_parameters):
#    #COME BACK THIS IS WRONG!!!
#    #This is the function that is integrated to compute E[F(x)]
#    #x is the point at which the function is evaluated
#    #suffstat_index is the index of the suffstat that is being integrated over
#    #test_parameters is of type test_class.TestParameters.
#    #The function returns the value of the function at x.
#    return test_parameters.F(x) * test_parameters.suffstats[suffstat_index].get_coefficient(x)


def mle_direct_covariance(test_parameters):
    #test_parameters is of type test_class.TestParameters.
    #The formula for the Fisher matrix is E[F(x) F(x)^T] - E[F(x)] E[F(x)^T].
    #The covariance should then be the inverse of the Fisher matrix.
    #Note that because F appears twice in each term, it is okay to use
    #the negative of F instead. This is important because my code currently
    #uses the convention that sufficient statistics are non-positive
    #and the coefficients are non-negative, as opposed to the LaTeX document
    #which has the opposite convention.
    
    #First, compute the normalization
    def non_normalized(x, test_parameters):
        stat_vals = [suffstat.zeroth_derivative(np.array([x])) for suffstat in test_parameters.suffstats]
        exponent = np.dot(test_parameters.theta_star, np.array(stat_vals))[0]
        return np.exp(exponent)

    #First, compute E[F(x)]
    E_Fx = []
    for suffstat in test_parameters.suffstats:

        E_Fx.append(quad(test_parameters.F, test_parameters.lower_bound[i], test_parameters.upper_bound[i])[0])
    E_Fx = np.array(E_Fx)
