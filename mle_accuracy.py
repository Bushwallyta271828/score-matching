import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import sufficient_statistics
import sample
#import matplotlib.pyplot as plt


def non_normalized_pdf(x, thetas):
    return np.exp(sufficient_statistics.sufficient_statistics(np.array([x]), thetas))


def normalize(thetas):
    #compute the partition function
    return quad(non_normalized_pdf, -np.infty, np.infty, args=(thetas,))[0]


def negative_log_likelihood(thetas, xs):
    #assert all thetas are positive:
    if np.any(thetas <= 0):
        raise ValueError("thetas <= 0")
    Z = normalize(thetas)
    log_likelihood = np.sum(sufficient_statistics.sufficient_statistics(xs, thetas) - np.log(Z))
    return -log_likelihood


def mle_accuracy(n, theta_star, initial_theta):
    #assert all thetas are positive:
    if np.any(theta_star <= 0):
        raise ValueError("theta_star <= 0")
    if np.any(initial_theta <= 0):
        raise ValueError("initial_theta <= 0")
    #find the samples:
    samples = sample.sample(theta_star, n)
    #find the MLE with theta bounded to be positive:
    bounds = [(0, None) for i in range(np.size(theta_star))]
    MLE_answer = minimize(negative_log_likelihood, initial_theta, args=(samples,), bounds=bounds).x
    dist = np.sqrt(np.sum((MLE_answer - theta_star)**2))
    return (MLE_answer, dist)


#xs = []
#ys = []
#for i in range(1,21):
#    n = 10000*i
#    print("n =", n)
#    dist = mle_accuracy(n, np.array([1.5, 0.5]), np.array([1, 1]))[1]
#    xs.append(n)
#    ys.append(dist)
#plt.plot(xs, ys, '.')
#plt.show()
