from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from import_csv import load_data_from_csv

input_training_csv = 'credit_card_training.csv'
input_test_csv = 'credit_card_test.csv'
training_feature_names, training_instances, training_labels = load_data_from_csv(input_csv=input_training_csv)
test_feature_names, test_instances, test_labels = load_data_from_csv(input_csv=input_test_csv)

for x in range(1, 20):
    classifier = KNeighborsClassifier(n_neighbors=x)
    classifier.fit(training_instances, training_labels)
    predicted_test_labels = classifier.predict(test_instances)

    print("Neighbors = ", x)
    print(classification_report(test_labels, predicted_test_labels, digits=3))
