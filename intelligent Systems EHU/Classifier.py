from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from import_csv import load_data_from_csv

from sklearn.metrics import *

class Classifier(object):
    """description of class"""

    def __init__(self, training, test):
        self.training_feature_names, self.training_instances, self.training_labels = load_data_from_csv(input_csv=training)
        self.test_feature_names, self.test_instances, self.test_labels = load_data_from_csv(input_csv=test)

    def predict(self, classifier, export=True):
        classifier.fit(self.training_instances, self.training_labels)
        predicted_test_labels = classifier.predict(self.test_instances)
        if export:
            print(confusion_matrix(self.test_labels, predicted_test_labels))
            print(Classifier.average_report(self.test_labels, predicted_test_labels, digits=3))
        else:
            print(classification_report(self.test_labels, predicted_test_labels, digits=3))
        
    def average_report(y_true, y_pred, digits):
        p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
        # compute averages
        return (u'{:>9.{digits}f}' * 3).format(np.average(p, weights=s), 
                              np.average(r, weights=s),
                              np.average(f1, weights=s),
                              digits=digits)


