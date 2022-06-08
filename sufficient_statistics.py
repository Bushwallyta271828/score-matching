import numpy as np

#Note: I will assume that the family of distributions is of the form
#Z^(-1) exp(thetas[0] * (-x^2 / 2) + thetas[1] * later_suff_stats[0](x) + ...)
#where later_suff_stats[i](x) is everywhere non-positive.
#This form is necessary for rejection sampling.

def later_sufficient_statistics(xs, later_thetas):
    #later_thetas is a column vector
    #xs is a column vector
    later_suff_stats = np.array([-xs*xs*xs*xs])
    return np.dot(np.transpose(later_suff_stats), later_thetas)


def sufficient_statistics(xs, thetas):
    #thetas is a column vector
    #xs is a column vector
    gaussian = thetas[0] * (-xs*xs / 2)
    later = later_sufficient_statistics(xs, thetas[1:])
    return gaussian + later


def first_derivatives(xs):
    #works because xs is one-dimensional.
    #used for scorematching
    return np.array([-xs, -4*xs*xs*xs])


def second_derivatives(xs):
    #works because xs is one-dimensional.
    #used for scorematching
    return np.array([-1*np.ones(xs.shape), -12*xs*xs])
