from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from mlxtend.classifier import StackingClassifier

import numpy

from multiprocessing import Pool

from Classifier import Classifier

CONNECT4_TRAINING = 'data/connect4/connect-4.training.csv'
CONNECT4_TEST = 'data/connect4/connect-4.test.csv'
TIC_TRAINING = 'data/tic_tac_toe/tic-tac-toe.training.csv'
TIC_TEST = 'data/tic_tac_toe/tic-tac-toe.test.csv'
REVIEWS_TRAINING = 'data/reviews/reviews_Video_Games_training.csv'
REVIEWS_TEST = 'data/reviews/reviews_Video_Games_test.csv'

def execute(classifier, task=0):
    if task == 0:
        for x in range(1, 51):
            classifier.predict(KNeighborsClassifier(x, n_jobs=8))
    elif task == 1:
        classifier.predict(tree.DecisionTreeClassifier())
    elif task == 2:
        for x in numpy.arange(10, 301, 10):
            classifier.predict(RandomForestClassifier(x, n_jobs=8, max_features='auto'))
    elif task == 3:
        classifier.predict(MLPClassifier(hidden_layer_sizes=(1000,), max_iter=5000000, solver='lbfgs'))
    elif task == 4:
        classifier.predict(MLPClassifier(hidden_layer_sizes=(1000,), max_iter=5000000, solver='adam'))
    elif task == 5:
        classifier.predict(svm.SVC(kernel='linear'), False)
    elif task == 6:
        classifier.predict(svm.SVC(kernel='rbf'), False)

def stacking(classifier, optimalRF, optimalKNN, task=0):
    if task == 0:
        for x in range(1, 51):
            classifier.predict(StackingClassifier(
                classifiers=[RandomForestClassifier(optimalRF, n_jobs=8), KNeighborsClassifier(optimalKNN, n_jobs=8)], 
                meta_classifier=KNeighborsClassifier(x, n_jobs=8)))
    elif task == 2:
        classifier.predict(StackingClassifier(
            classifiers=[RandomForestClassifier(optimalRF, n_jobs=8), KNeighborsClassifier(optimalKNN, n_jobs=8)], 
            meta_classifier=tree.DecisionTreeClassifier()))
    elif task == 5:
        classifier.predict(StackingClassifier(
            classifiers=[RandomForestClassifier(optimalRF, n_jobs=8), KNeighborsClassifier(optimalKNN, n_jobs=8)], 
            meta_classifier=svm.SVC(kernel='linear')))
    elif task == 6:
        classifier.predict(StackingClassifier(
            classifiers=[RandomForestClassifier(optimalRF, n_jobs=8), KNeighborsClassifier(optimalKNN, n_jobs=8)], 
            meta_classifier=svm.SVC(kernel='rbf')))

def test_program(classifier):
    print('K-NN')
    execute(classifier, 0)
    print('DecisionTreeClassifier')
    execute(classifier, 1)
    #print('RandomForestClassifier')
    #execute(classifier, 2)
    #print('MLPClassifier (lbfgs)')
    #execute(classifier, 3)
    #print('MLPClassifier (adam)')
    #execute(classifier, 4)
    print('SVC (linear)')
    execute(classifier, 5)
    print('SVC (rbf)')
    execute(classifier, 6)

def test_program_stacking(classifier, optimalRF, optimalKNN):
    print('StackingClassifier (KNeighborsClassifier)')
    stacking(classifier, optimalRF, optimalKNN, 0)
    print('StackingClassifier (DecisionTreeClassifier)')
    stacking(classifier, optimalRF, optimalKNN, 2)
    print('StackingClassifier (SVC linear)')
    stacking(classifier, optimalRF, optimalKNN, 5)
    print('StackingClassifier (SVC rbf)')
    stacking(classifier, optimalRF, optimalKNN, 6)

if __name__ == '__main__':
    cl = Classifier(REVIEWS_TRAINING, REVIEWS_TEST)
    test_program(cl);

    #test_program_stacking(cl, 90, 10) # 110, 5
    #stacking(cl, 110, 5, 5)

    ## CONNECT4_TRAINING, CONNECT4_TEST
    #cl.predict(RandomForestClassifier(110, n_jobs=8, max_features='auto'))
    #cl.predict(KNeighborsClassifier(5, n_jobs=8))