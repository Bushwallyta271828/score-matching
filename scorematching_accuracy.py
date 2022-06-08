import numpy as np
from numpy.linalg import inv
import sufficient_statistics
import sample
#import matplotlib.pyplot as plt


def scorematching_accuracy(n, theta_star):
    #assert all thetas are positive:
    if np.any(theta_star <= 0):
        raise ValueError("theta_star <= 0")
    samples = sample.sample(theta_star, n)

    first = sufficient_statistics.first_derivatives(samples).T
    matrix_term_of_x = first[:,:,np.newaxis] * first[:,np.newaxis,:]
    matrix_term = np.average(matrix_term_of_x, axis=0)
    second = sufficient_statistics.second_derivatives(samples).T
    vector_term = np.average(second, axis=0)

    scorematch_answer = -np.dot(inv(matrix_term), vector_term)
    dist = np.sqrt(np.sum((scorematch_answer - theta_star)**2))
    return (scorematch_answer, dist)


#xs = []
#ys = []
#for i in range(1,21):
#    print(i)
#    n = 100000*i
#    dist = scorematching_accuracy(n, np.array([1.5, 0.5]))[1]
#    xs.append(n)
#    ys.append(dist)
#plt.plot(xs, ys, '.')
#plt.show()
