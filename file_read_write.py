import numpy as np
import os


def write_to_file(methods, ns, runs, theta_stars, accuracies, means, covs, test_number=None):
    if test_number == None:
        #look in the directory ./results/ for the highest numbered file
        #and use that number + 1 for the test number
        #files are of the form test_number_i.txt
        test_number = 1
        for file in os.listdir('./results/'):
            if file.startswith('test_number_'):
                if file.split('_')[2].split('.')[0].isdigit():
                    file_number = int(file.split('_')[2].split('.')[0])
                    test_number = max(file_number + 1, test_number)
    filename = './results/test_number_' + str(test_number) + '.txt'
    f = open(filename, 'w')
    for i in range(len(methods)):
        f.write("Data point: " + str(i) + '\n')
        f.write("method: " + str(methods[i]) + '\n')
        f.write("n: " + str(ns[i]) + '\n')
        f.write("runs: " + str(runs[i]) + '\n')
        f.write("theta_star: " + str(theta_stars[i].tolist()) + '\n')
        f.write("accuracy: " + str(accuracies[i]) + '\n')
        f.write("mean: " + str(means[i].tolist()) + '\n')
        f.write("cov: " + str(covs[i].tolist()) + '\n')
        f.write('\n')
    f.close()


def read_from_file(test_number=None):
    if test_number == None:
        #look in the directory ./results/ for the highest numbered file
        #files are of the form test_number_i.txt
        for file in os.listdir('./results/'):
            if file.startswith('test_number_'):
                if file.split('_')[2].split('.')[0].isdigit():
                    file_number = int(file.split('_')[2].split('.')[0])
                    test_number = max(file_number, test_number)
    filename = './results/test_number_' + str(test_number) + '.txt'
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    methods, ns, runs, theta_stars, accuracies, means, covs = [], [], [], [], [], [], []
    for line in lines:
        if line.startswith('method:'):
            methods.append(line.split(':')[1].strip())
        elif line.startswith('n:'):
            ns.append(int(line.split(':')[1].strip()))
        elif line.startswith('runs:'):
            runs.append(int(line.split(':')[1].strip()))
        elif line.startswith('theta_star:'):
            theta_stars.append(np.array(eval(line.split(':')[1].strip())))
        elif line.startswith('accuracy:'):
            accuracies.append(float(line.split(':')[1].strip()))
        elif line.startswith('mean:'):
            means.append(np.array(eval(line.split(':')[1].strip())))
        elif line.startswith('cov:'):
            covs.append(np.array(eval(line.split(':')[1].strip())))
    return methods, ns, runs, theta_stars, accuracies, means, covs

