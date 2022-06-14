import numpy as np


#Note: I will assume that the family of distributions is of the form
#Z^(-1) exp(thetas[0] * (-x^2 / 2) + thetas[1] * later_suff_stats[0](x) + ...)
#where later_suff_stats[i](x) is everywhere non-positive.
#This form is necessary for rejection sampling.


class SuffStatParams:
    def __init__(self):
        pass


class PolyParams(SuffStatParams):
    def __init__(self, exponent):
        self.exponent = exponent


class SinusoidParams(SuffStatParams):
    def __init__(self, amplitude, frequency, phase):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase


def later_sufficient_statistics(xs, later_thetas, params):
    #later_thetas is a column vector
    #xs is a column vector
    #params is a list of objects of type SuffStatParams
    later_suff_stats_list = []
    for i in range(len(params)):
        if isinstance(params[i], PolyParams):
            later_suff_stat = -xs**params[i].exponent
        elif isinstance(params[i], SinusoidParams):
            raise NotImplementedError
            #later_suff_stat = params[i].amplitude * (np.sin(params[i].frequency * xs + params[i].phase) - 1)
        else:
            raise ValueError("params[i] is not a recognized type")
        later_suff_stats_list.append(later_suff_stat)
    later_suff_stats = np.array(later_suff_stats_list)
    return np.dot(np.transpose(later_suff_stats), later_thetas)


def later_first_derivatives(xs, params):
    #works because xs is one-dimensional.
    #used for scorematching


def later_second_derivatives(xs, params):
    #works because xs is one-dimensional.
    #used for scorematching


def sufficient_statistics(xs, thetas, params):
    #thetas is a column vector
    #xs is a column vector
    #params is a list of objects of type SuffStatParams
    gaussian = thetas[0] * (-xs*xs / 2)
    later = later_sufficient_statistics(xs, thetas[1:], params)
    return gaussian + later


def first_derivatives(xs):
    #works because xs is one-dimensional.
    #used for scorematching
    return np.array([-xs, -4*xs*xs*xs])


def second_derivatives(xs):
    #works because xs is one-dimensional.
    #used for scorematching
    return np.array([-1*np.ones(xs.shape), -12*xs*xs])
