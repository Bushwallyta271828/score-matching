import numpy as np
import test_class
import os
import pickle


def write_to_file(tests, test_number=None):
    #tests is a list of test_class.Test objects
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
    #write the test results to a file using pickle
    f = open(filename, 'wb')
    pickle.dump(tests, f)
    f.close()


def read_from_file(test_number=None):
    #returns a list of test_class.Test objects
    if test_number == None:
        #look in the directory ./results/ for the highest numbered file
        #files are of the form test_number_i.txt
        for file in os.listdir('./results/'):
            if file.startswith('test_number_'):
                if file.split('_')[2].split('.')[0].isdigit():
                    file_number = int(file.split('_')[2].split('.')[0])
                    test_number = max(file_number, test_number)
    filename = './results/test_number_' + str(test_number) + '.txt'
    #read the test results from a file using pickle
    f = open(filename, 'rb')
    tests = pickle.load(f)
    f.close()
    return tests
