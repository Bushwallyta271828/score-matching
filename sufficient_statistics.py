import numpy as np


#Note: I will assume that the family of distributions is of the form
#Z^(-1) exp(thetas[0] * (-x^2 / 2) + thetas[1] * later_suff_stats[0](x) + ...)
#where later_suff_stats[i](x) is everywhere non-positive.
#This form is necessary for rejection sampling.


class SuffStat:
    def __init__(self, zeroth_derivative, first_derivative, second_derivative):
        self.zeroth_derivative = zeroth_derivative
        self.first_derivative = first_derivative
        self.second_derivative = second_derivative


class FirstStat(SuffStat):
    def __init__(self):
        pass

    def zeroth_derivative(self, xs):
        return -xs**2 / 2

    def first_derivative(self, xs):
        return -xs

    def second_derivative(self, xs):
        return -np.ones(xs.shape)


class PolyStat(SuffStat):
    def __init__(self, exponent):
        self.exponent = exponent

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

    def zeroth_derivative(self, xs):
        return self.amplitude * (np.sin(self.frequency * xs + self.phase) - 1)

    def first_derivative(self, xs):
        return self.amplitude * self.frequency * np.cos(self.frequency * xs + self.phase)

    def second_derivative(self, xs):
        return -self.amplitude * self.frequency**2 * np.sin(self.frequency * xs + self.phase)
