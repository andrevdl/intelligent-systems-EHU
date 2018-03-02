from import_csv import load_data_from_csv
from sklearn.metrics import classification_report

def _ensemble(input_training_csv, input_test_csv, estimators, voting):
    # load training dataset
    _, training_instances, training_labels = load_data_from_csv(input_csv=input_training_csv)
    # load test dataset
    _, test_instances, test_labels = load_data_from_csv(input_csv=input_test_csv)
    # define Majority Vote Classifier
    from sklearn.ensemble import VotingClassifier
    voting_classifier = VotingClassifier(estimators=estimators, voting=voting)
    # train Majority Vote Classifier
    voting_classifier.fit(training_instances, training_labels)
    # get predicted labels
    predicted_test_labels = voting_classifier.predict(test_instances)
    # compute evaluation scores
    print(classification_report(test_labels, predicted_test_labels, digits=3))

def majority_vote(input_training_csv, input_test_csv, estimators):
    _ensemble(input_training_csv, input_test_csv, estimators, 'hard')

def confidence_based_ensemble(input_training_csv, input_test_csv, estimators):
    _ensemble(input_training_csv, input_test_csv, estimators, 'soft')
