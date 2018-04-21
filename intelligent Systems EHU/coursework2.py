import sklearn
from sklearn.feature_extraction.text import *
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import pandas as pd
from pandas import *
import numpy as np
import re

# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# http://scikit-learn.org/stable/modules/feature_extraction.html#limitations-of-the-bag-of-words-representation

class LexiconPolarity(BaseEstimator, TransformerMixin):
    """ Polarity featute based on bag-of-words and a sentimental lexicon
    Lexicon Feature value can use the actually values of the words (binary=False) or
    translate the values to -1 for negative words and +1 for positive words (Binary=True).

    The sum of the words is the value of the lexicon feature.

    By the parameter enabled can the feature be enabled or disabled.
    In the case of disable, it returns 0 for each sample.
    """
    def __init__(self, lexicon, binary=False, enabled=True):
        if not isinstance(lexicon, dict):
            raise ValueError('Lexicon is not a dict')

        self.lexicon = lexicon
        self.binary = binary
        self.enabled = enabled

    def _tokenizer(self):
        """ Create a function to tokenize text to tokens
        The returned function is able to tokenize text to tokens.
        A single token is a single word.
        """
        pattern = re.compile('(?u)\\b\\w\\w+\\b')
        return lambda doc: pattern.findall(doc)

    def transform(self, X, **transform_params):
        """ Transform the text from each sample to a lexicon feature 
        """

        result = []

        if not self.enabled:
            for x in X:
                result.append(0.0)
            return np.array(result).reshape(-1, 1)

        tokenize = self._tokenizer()
        lexicon = self.lexicon
        binary = self.binary
        
        for x in X:
            score = 0.0
            words = tokenize(x)
            for w in words:
                s = lexicon.get(w)
                if s != None and not binary:
                    score += s
                elif s != None and binary:
                    score += np.sign(s)

            result.append(score)

        return np.array(result).reshape(-1, 1)

    def fit(self, X, y=None, **fit_params):
        return self

if __name__ == '__main__':
    VERBOSE = 5 # default = 0

    # Loading the sentimental lexicon and transform it to a dict
    lexicon_table = pd.read_table('data/raw/Games_senti_lexicon.tsv', header=None)
    lexicon_dict = dict(zip(list(lexicon_table[0]), list(lexicon_table[1])))

    # TF-(IDF) and Random Forest classifiers
    tfidfVect = TfidfVectorizer(analyzer='word', stop_words='english', max_features=2000)
    randomForest = RandomForestClassifier(n_jobs=4)

    # Process pipeline
    # Creating the features and load it into the classifier
    pipe = Pipeline([
        (
            'features', 
            FeatureUnion([
                ('vect', CountVectorizer(analyzer='word', stop_words='english', max_features=2000)), 
                ('lex', Pipeline([('polarity', LexiconPolarity(lexicon=lexicon_dict)), ('scalar', MinMaxScaler())]))
            ])
        ), 
        ('classifier', MultinomialNB())
    ])

    # Definitions of all the runs
    #
    # Runs are defined in sets.
    # Each set has custom parameters, that may override the defaults.
    # All the parameters in the set are cross joined and each cross join is a run.
    # Total amount of runs is the sum of sets, where each set is a cross join of all the parameters in a set.
    param_grid = [
        {
            'features__vect__ngram_range': ((1, 1), (1, 2), (2, 2)),
            'features__vect__binary': (True, False),
            'features__lex__polarity__binary': (True, False),
            'features__lex__polarity__enabled': (True, False)
        },
        {
            'features__vect': [tfidfVect],
            'features__vect__ngram_range': ((1, 1), (1, 2), (2, 2)),
            'features__vect__use_idf': (True, False),
            'features__lex__polarity__binary': (True, False),
            'features__lex__polarity__enabled': (True, False)
        },
        {
            'features__vect__ngram_range': ((1, 1), (1, 2), (2, 2)),
            'features__vect__binary': (True, False),
            'features__lex__polarity__binary': (True, False),
            'features__lex__polarity__enabled': (True, False),
            'classifier': [SVC()],
            'classifier__kernel': ('linear', 'rbf')
        },
        {
            'features__vect': [tfidfVect],
            'features__vect__ngram_range': ((1, 1), (1, 2), (2, 2)),
            'features__vect__use_idf': (True, False),
            'features__lex__polarity__binary': (True, False),
            'features__lex__polarity__enabled': (True, False),
            'classifier': [SVC()],
            'classifier__kernel': ('linear', 'rbf')
        },
        {
            'features__vect__ngram_range': ((1, 1), (1, 2), (2, 2)),
            'features__vect__binary': (True, False),
            'features__lex__polarity__binary': (True, False),
            'features__lex__polarity__enabled': (True, False),
            'classifier': [randomForest],
            'classifier__n_estimators': (np.arange(10, 301, 10))
        },
        {
            'features__vect': [tfidfVect],
            'features__vect__ngram_range': ((1, 1), (1, 2), (2, 2)),
            'features__vect__use_idf': (True, False),
            'features__lex__polarity__binary': (True, False),
            'features__lex__polarity__enabled': (True, False),
            'classifier': [randomForest],
            'classifier__n_estimators': (np.arange(10, 301, 10))
        }
    ]

    # Creating classification object, that is capable to create a report from all the runs.
    # The runs are defined in the param_grid
    # All the runs are perfomed in 3 fold cross validation.
    grid = GridSearchCV(pipe, param_grid, n_jobs=8, return_train_score=False, verbose=VERBOSE)

    # Reading the dataset with the reviews and put it into a table (Merged dataset of the traing and test set)
    reviews = pd.read_table('data/raw/reviews_Video_Games_merged.raw.tsv', names=['content', 'label'])

    # Execute the GridSearchCV and classify all the runs in 3 fold cross validation
    grid.fit(reviews['content'], reviews['label'])

    # Saving the GridSearchCV report in to a CSV file.
    df = DataFrame(grid.cv_results_)
    df.to_csv('results.csv', sep=';', float_format='%.3f', decimal=',')