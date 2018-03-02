from sklearn import svm
from sklearn.metrics import classification_report

from import_csv import load_data_from_csv

training_feature_names, training_instances, training_labels = load_data_from_csv(input_csv='data/tic_tac_toe/tic-tac-toe.training.csv')
test_feature_names, test_instances, test_labels = load_data_from_csv(input_csv='data/tic_tac_toe/tic-tac-toe.test.csv')

classifier = svm.SVC(kernel='rbf')
classifier.fit(X=training_instances, y=training_labels)
predicted_test_labels = classifier.predict(test_instances)

print(classification_report(test_labels, predicted_test_labels, digits=3))
