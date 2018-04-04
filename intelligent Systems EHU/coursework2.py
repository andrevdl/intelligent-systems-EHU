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

#import matplotlib.pyplot as plt

import pandas as pd
from pandas import *
import numpy as np
import re
import json

# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# http://scikit-learn.org/stable/modules/feature_extraction.html#limitations-of-the-bag-of-words-representation

class LexiconPolarity(BaseEstimator, TransformerMixin):
    def __init__(self, lexicon, binary=False, enabled=True):
        if not isinstance(lexicon, dict):
            raise ValueError('Lexicon is not a dict')

        self.lexicon = lexicon
        self.binary = binary
        self.enabled = enabled

    def _tokenizer(self):
        pattern = re.compile('(?u)\\b\\w\\w+\\b')
        return lambda doc: pattern.findall(doc)

    def transform(self, X, **transform_params):
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
    lexicon_table = pd.read_table('data/raw/Games_senti_lexicon.tsv', header=None)
    lexicon_dict = dict(zip(list(lexicon_table[0]), list(lexicon_table[1])))

    tfidfVect = TfidfVectorizer(analyzer='word', stop_words='english', max_features=2000)
    randomForest = RandomForestClassifier(n_jobs=4)

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

    grid = GridSearchCV(pipe, param_grid, n_jobs=8, return_train_score=False, verbose=5) #, cv=10)

    reviews = pd.read_table('data/raw/reviews_Video_Games_merged.raw.tsv', names=['content', 'label']) # training and then test
    grid.fit(reviews['content'], reviews['label'])

    df = DataFrame(grid.cv_results_)
    df.to_csv('results.csv', sep=';', float_format='%.3f', decimal=',')

    #X_train, X_test, y_train, y_test = train_test_split(reviews['content'], reviews['label'], test_size=0.5, random_state=0);