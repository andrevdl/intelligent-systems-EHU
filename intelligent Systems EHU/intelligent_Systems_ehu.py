from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from mlxtend.classifier import StackingClassifier

import numpy

from multiprocessing import Pool

from Classifier import Classifier

CONNECT4_TRAINING = 'data/connect4/connect-4.training.csv';
CONNECT4_TEST = 'data/connect4/connect-4.test.csv';
TIC_TRAINING = 'data/tic_tac_toe/tic-tac-toe.training.csv';
TIC_TEST = 'data/tic_tac_toe/tic-tac-toe.test.csv';

def execute(classifier, task=0):
    if task == 0:
        for x in range(1, 51):
            cl.predict(KNeighborsClassifier(x, n_jobs=8))
    elif task == 1:
        cl.predict(tree.DecisionTreeClassifier())
    elif task == 2:
        for x in numpy.arange(10, 301, 10):
            cl.predict(RandomForestClassifier(x, n_jobs=8, max_features='auto'))
    elif task == 3:
        cl.predict(MLPClassifier(hidden_layer_sizes=(1000,), max_iter=5000000, solver='lbfgs'))
    elif task == 4:
        cl.predict(MLPClassifier(hidden_layer_sizes=(1000,), max_iter=5000000, solver='adam'))
    elif task == 5:
        cl.predict(AdaBoostClassifier(base_estimator=RandomForestClassifier(40, n_jobs=8), n_estimators=5000))

def stacking(classifier, optimalRF, optimalKNN, task=0):
    if task == 0:
        for x in range(1, 51):
            cl.predict(StackingClassifier(
                classifiers=[RandomForestClassifier(optimalRF, n_jobs=8), KNeighborsClassifier(optimalKNN)], 
                meta_classifier=KNeighborsClassifier(x, n_jobs=8)))
    elif task == 1:
        for x in numpy.arange(10, 301, 10):
            cl.predict(StackingClassifier(
                classifiers=[RandomForestClassifier(optimalRF, n_jobs=8), KNeighborsClassifier(optimalKNN)], 
                meta_classifier=RandomForestClassifier(x, n_jobs=8)))
    elif task == 2:
        cl.predict(StackingClassifier(
            classifiers=[RandomForestClassifier(optimalRF, n_jobs=8), KNeighborsClassifier(optimalKNN)], 
            meta_classifier=tree.DecisionTreeClassifier()))

def test_program(classifier):
    print('K-NN')
    execute(classifier, 0)
    print('DecisionTreeClassifier')
    execute(classifier, 1)
    print('RandomForestClassifier')
    execute(classifier, 2)
    print('MLPClassifier (lbfgs)')
    execute(classifier, 3)
    print('MLPClassifier (adam)')
    execute(classifier, 4)
    print('AdaBoostClassifier')
    execute(classifier, 5)

def test_program_stacking(classifier, optimalRF, optimalKNN):
    print('StackingClassifier (KNeighborsClassifier)')
    stacking(classifier, optimalRF, optimalKNN, 0)
    print('StackingClassifier (RandomForestClassifier)')
    stacking(classifier, optimalRF, optimalKNN, 1)
    print('StackingClassifier (DecisionTreeClassifier)')
    stacking(classifier, optimalRF, optimalKNN, 2)

if __name__ == '__main__':
    cl = Classifier(CONNECT4_TRAINING, CONNECT4_TEST)
    #test_program(cl)
    execute(cl, 4)
