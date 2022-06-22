import numpy as np
from numpy.linalg import inv
import sufficient_statistics
import test_class
import sample


def scorematching(test_parameters):
    #test is of type test_class.TestParameters

    #assert all thetas are positive:
    if np.any(test_parameters.theta_star <= 0):
        raise ValueError("theta_star <= 0")
    samples = sample.sample(test_parameters.theta_star, test_parameters.n)

    first = sufficient_statistics.first_derivatives(test_parameters.suffstats, samples).T
    matrix_term_of_x = first[:,:,np.newaxis] * first[:,np.newaxis,:]
    matrix_term = np.average(matrix_term_of_x, axis=0)
    second = sufficient_statistics.second_derivatives(test_parameters.suffstats, samples).T
    vector_term = np.average(second, axis=0)

    scorematch_answer = -np.dot(inv(matrix_term), vector_term)
    dist = np.sqrt(np.sum((scorematch_answer - theta_star)**2))
    return (scorematch_answer, dist)

