import numpy as np
from sufficient_statistics import SuffStatParameters


class TestParameters:
    def __init__(self, suffparams, theta_star, n, method, runs):
        self.suffparams = suffparams #list of SuffStatParameters
        self.theta_star = theta_star #1d np.array
        self.n = n #int
        self.method = method #string
        self.runs = runs #int

    #def __str__(self):
    #    s = "method: " + self.method + "\n"
    #    s += "n: " + str(self.n) + "\n"
    #    s += "runs: " + str(self.runs) + "\n"
    #    s += "theta_star: " + str(self.theta_star.tolist()) + "\n"
    #    return s

    #def from_string(self, s):
    #    #Note: can be given more information, i.e.
    #    #can be passed string representation
    #    #of entire TestClass
    #    split_s = [x.split(": ") for x in s.split("\n")]
    #    for ss in split_s:
    #        if ss[0] == "method":
    #            self.method = ss[1]
    #        elif ss[0] == "n":
    #            self.n = int(ss[1])
    #        elif ss[0] == "runs":
    #            self.runs = int(ss[1])
    #        elif ss[0] == "theta_star":
    #            self.theta_star = np.array(eval(ss[1]))


class TestResults:
    def __init__(self, accuracy, mean, cov):
        self.accuracy = accuracy #float
        self.mean = mean #1d np.array
        self.cov = cov #2d np.array

    #def __str__(self):
    #    s = "accuracy: " + str(self.accuracy) + "\n"
    #    s += "mean: " + str(self.mean.tolist()) + "\n"
    #    s += "cov: " + str(self.cov.tolist()) + "\n"
    #    return s

    #def from_string(self, s):
    #    #Note: can be given more information, i.e.
    #    #can be passed string representation
    #    #of entire TestClass
    #    split_s = [x.split(": ") for x in s.split("\n")]
    #    for ss in split_s:
    #        if ss[0] == "accuracy":
    #            self.accuracy = float(ss[1])
    #        elif ss[0] == "mean":
    #            self.mean = np.array(eval(ss[1]))
    #        elif ss[0] == "cov":
    #            self.cov = np.array(eval(ss[1]))


class Test:
    def __init__(self, parameters, results):
        self.parameters = parameters #instance of TestParameters
        self.results = results #instance of TestResults

    #def __str__(self):
    #    return str(self.parameters) + str(self.results)

    #def from_string(self, s):
    #    self.parameters = TestParametersClass.from_string(s)
    #    self.results = TestResultsClass.from_string(s)
