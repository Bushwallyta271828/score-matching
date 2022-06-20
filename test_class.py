import numpy as np
import sufficient_statistics as ss


class TestParameters:
    def __init__(self, suffstats, theta_star, n, method, runs):
        self.suffstats = suffstats #list of SuffStat objects
        self.theta_star = theta_star #1d np.array
        self.n = n #int
        self.method = method #string
        self.runs = runs #int


class TestResults:
    def __init__(self, accuracy, mean, cov):
        self.accuracy = accuracy #float
        self.mean = mean #1d np.array
        self.cov = cov #2d np.array


class Test:
    def __init__(self, parameters, results):
        self.parameters = parameters #instance of TestParameters
        self.results = results #instance of TestResults

