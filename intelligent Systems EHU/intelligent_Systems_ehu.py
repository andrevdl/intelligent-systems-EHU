from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from mlxtend.classifier import StackingClassifier

import numpy

from multiprocessing import Pool

import ensemble
from import_csv import load_data_from_csv

CONNECT4_TRAINING = 'data/connect4/connect-4.training.csv';
CONNECT4_TEST = 'data/connect4/connect-4.test.csv';
TIC_TRAINING = 'data/tic_tac_toe/tic-tac-toe.training.csv';
TIC_TEST = 'data/tic_tac_toe/tic-tac-toe.test.csv';

CURRENT_TRAINING = CONNECT4_TRAINING;
CURRENT_TEST = CONNECT4_TEST;

def execute_majority_vote(estimators):
    ensemble.majority_vote(
        CONNECT4_TRAINING,
        CONNECT4_TEST,
        estimators
    )

def execute_confidence_based(estimators):
    ensemble.confidence_based_ensemble(
        CONNECT4_TRAINING,
        CONNECT4_TEST,
        estimators
    )

def KNN(n):
    training_feature_names, training_instances, training_labels = load_data_from_csv(input_csv=CURRENT_TRAINING)
    test_feature_names, test_instances, test_labels = load_data_from_csv(input_csv=CURRENT_TEST)

    classifier = KNeighborsClassifier(n_neighbors=n)
    classifier.fit(training_instances, training_labels)
    predicted_test_labels = classifier.predict(test_instances)

    print("Neighbors = ", n)
    print(classification_report(test_labels, predicted_test_labels, digits=3))

def DecisionTree():
    training_feature_names, training_instances, training_labels = load_data_from_csv(input_csv=CURRENT_TRAINING)
    test_feature_names, test_instances, test_labels = load_data_from_csv(input_csv=CURRENT_TEST)

    classifier = tree.DecisionTreeClassifier()
    classifier.fit(training_instances, training_labels)
    predicted_test_labels = classifier.predict(test_instances)

    print(classification_report(test_labels, predicted_test_labels, digits=3))

def RandomForest(e, f):
    training_feature_names, training_instances, training_labels = load_data_from_csv(input_csv=CURRENT_TRAINING)
    test_feature_names, test_instances, test_labels = load_data_from_csv(input_csv=CURRENT_TEST)

    classifier = RandomForestClassifier(e, n_jobs=8, max_features=f)
    classifier.fit(training_instances, training_labels)
    predicted_test_labels = classifier.predict(test_instances)

    print(classification_report(test_labels, predicted_test_labels, digits=3))

def Neutral(solver):
    training_feature_names, training_instances, training_labels = load_data_from_csv(input_csv=CURRENT_TRAINING)
    test_feature_names, test_instances, test_labels = load_data_from_csv(input_csv=CURRENT_TEST)

    classifier = MLPClassifier(hidden_layer_sizes=(1000,), max_iter=5000000, solver=solver);
    classifier.fit(training_instances, training_labels)
    predicted_test_labels = classifier.predict(test_instances)

    print(classification_report(test_labels, predicted_test_labels, digits=3))

def AdaBoost():
    training_feature_names, training_instances, training_labels = load_data_from_csv(input_csv=CURRENT_TRAINING)
    test_feature_names, test_instances, test_labels = load_data_from_csv(input_csv=CURRENT_TEST)

    classifier = AdaBoostClassifier(base_estimator=RandomForestClassifier(40, n_jobs=8), n_estimators=5000)
    classifier.fit(training_instances, training_labels)
    predicted_test_labels = classifier.predict(test_instances)

    print(classification_report(test_labels, predicted_test_labels, digits=3))

def Stacking(classifiers, meta):
    training_feature_names, training_instances, training_labels = load_data_from_csv(input_csv=CURRENT_TRAINING)
    test_feature_names, test_instances, test_labels = load_data_from_csv(input_csv=CURRENT_TEST)

    classifier = StackingClassifier(classifiers=classifiers, meta_classifier=meta)
    classifier.fit(training_instances, training_labels)
    predicted_test_labels = classifier.predict(test_instances)

    print(classification_report(test_labels, predicted_test_labels, digits=3))

if __name__ == '__main__':
    #for x in range(25, 51): # 1 - 25
    #    KNN(x)

    #DecisionTree();

    #for x in numpy.arange(10, 301, 10):
    #    RandomForest(x, 'auto')

    Neutral('lbfgs');
    #Neutral('adam');

    #AdaBoost();

    #Stacking([RandomForestClassifier(110, n_jobs=8), KNeighborsClassifier(5)], KNeighborsClassifier(1))
    #Stacking([RandomForestClassifier(110, n_jobs=8), KNeighborsClassifier(5)], RandomForestClassifier(110, n_jobs=8))
    #Stacking([RandomForestClassifier(110, n_jobs=8), KNeighborsClassifier(5)], tree.DecisionTreeClassifier())

    # OLD STUFF
    #p = Pool(5)
    #p.map(KNN, range(1, 5))
    #p.map(execute_majority_vote, [estimators, estimators, estimators, estimators, estimators, estimators])
