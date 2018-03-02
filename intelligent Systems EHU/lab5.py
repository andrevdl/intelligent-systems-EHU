from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

import ensemble

classifier1 = KNeighborsClassifier(n_neighbors=1)
classifier2 = KNeighborsClassifier(n_neighbors=2)
classifier3 = tree.DecisionTreeClassifier()

estimators = [
    ('K-NN, N=1', classifier1),
    ('K-NN, N=2', classifier2),
    ('Decision tree', classifier3)]

ensemble.majority_vote(
    'data/credit_card_training.csv',
    'data/credit_card_test.csv',
    estimators
)

ensemble.confidence_based_ensemble(
    'data/credit_card_training.csv',
    'data/credit_card_test.csv',
    estimators
)