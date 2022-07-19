import numpy as np
from scipy.special import erf


#Note: I will assume that the family of distributions is of the form
#Z^(-1) exp(thetas[0] * (-x^2 / 2) + thetas[1] * later_suff_stats[0](x) + ...)
#where later_suff_stats[i](x) is everywhere non-positive.
#This form is necessary for rejection sampling.

#Note: the above note is depricated for pure-limit computations.


class SuffStat:
    def __init__(self, zeroth_derivative, first_derivative, second_derivative):
        self.zeroth_derivative = zeroth_derivative
        self.first_derivative = first_derivative
        self.second_derivative = second_derivative

    def __str__(self):
        pass


class FirstStat(SuffStat):
    def __init__(self):
        pass

    def __str__(self):
        return "FirstStat"

    def zeroth_derivative(self, xs):
        return -xs**2 / 2

    def first_derivative(self, xs):
        return -xs

    def second_derivative(self, xs):
        return -np.ones(xs.shape)


class PolyStat(SuffStat):
    def __init__(self, exponent):
        #assert(type(exponent) == int) #Note: problems because actually it's numpy.int32, mod requirement below largely covers this.
        assert(exponent % 2 == 0)
        assert(exponent >= 4) #can't be 2 because scorematching can't distinguish from FirstStat
        self.exponent = exponent

    def __str__(self):
        return "PolyStat(exponent={})".format(self.exponent)

    def zeroth_derivative(self, xs):
        return -xs**self.exponent

    def first_derivative(self, xs):
        return -self.exponent * xs**(self.exponent - 1)

    def second_derivative(self, xs):
        return -self.exponent * (self.exponent - 1) * xs**(self.exponent - 2)


class SinusoidStat(SuffStat):
    def __init__(self, amplitude, frequency, phase):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def __str__(self):
        return "SinusoidStat(amplitude={}, frequency={}, phase={})".format(self.amplitude, self.frequency, self.phase)

    def zeroth_derivative(self, xs):
        return self.amplitude * (np.sin(self.frequency * xs + self.phase) - 1)

    def first_derivative(self, xs):
        return self.amplitude * self.frequency * np.cos(self.frequency * xs + self.phase)

    def second_derivative(self, xs):
        return -self.amplitude * self.frequency**2 * np.sin(self.frequency * xs + self.phase)


class NormalizableSinusoidStat(SuffStat):
    #NOTE: This doesn't satisfy the <= 0 requirement!
    #Do not use this with non-limit computations!
    def __init__(self, omega):
        self.omega = omega

    def __str__(self):
        return "NormalizableSinusoidStat(omega={})".format(self.omega)

    def zeroth_derivative(self, xs):
        return -np.sin(self.omega * xs) - xs*xs / 2

    def first_derivative(self, xs):
        return -self.omega * np.cos(self.omega * xs) - xs

    def second_derivative(self, xs):
        return self.omega**2 * np.sin(self.omega * xs) - 1


class CenteredSinusoidStat(SuffStat):
    #NOTE: This doesn't satisfy the <= 0 requirement!
    #Do not use this with non-limit computations!
    def __init__(self, omega):
        self.omega = omega

    def __str__(self):
        return "CenteredSinusoidStat(omega={})".format(self.omega)

    def zeroth_derivative(self, xs):
        return -np.sin(self.omega * xs)

    def first_derivative(self, xs):
        return -self.omega * np.cos(self.omega * xs)

    def second_derivative(self, xs):
        return self.omega**2 * np.sin(self.omega * xs)


class BimodalPlusErfStat(SuffStat):
    def __init__(self, offset, erfcoeff):
        self.offset = offset
        self.erfcoeff = erfcoeff

    def __str__(self):
        return "BimodalPlusErfStat(offset={}, erfcoeff={})".format(self.offset, self.erfcoeff)

    def zeroth_derivative(self, xs):
        return xs**2 - xs**4 / (2 * self.offset**2) + self.erfcoeff * erf(xs)

    def first_derivative(self, xs):
        return 2 * xs - 2 * xs**3 / self.offset**2 + self.erfcoeff * 2 * np.exp(-xs**2) / np.sqrt(np.pi)

    def second_derivative(self, xs):
        return 2 - 6 * xs**2 / self.offset**2 + self.erfcoeff * (-4 * xs * np.exp(-xs**2) / np.sqrt(np.pi)) 


def zeroth_derivatives(suff_stats, xs):
    #suff_stats is a list of SuffStats
    #xs is a 1d numpy array
    return np.array([stat.zeroth_derivative(xs) for stat in suff_stats])

def first_derivatives(suff_stats, xs):
    #suff_stats is a list of SuffStats
    #xs is a 1d numpy array
    return np.array([stat.first_derivative(xs) for stat in suff_stats])

def second_derivatives(suff_stats, xs):
    #suff_stats is a list of SuffStats
    #xs is a 1d numpy array
    return np.array([stat.second_derivative(xs) for stat in suff_stats])

