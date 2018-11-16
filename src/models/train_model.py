# Standard Library Imports
from datetime import datetime
import os
# Third Party Imports
import pandas as pd
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Global Variables
DATA_INTERIM = os.environ['DATA_INTERIM']
DATA_PROCESSED = os.environ['DATA_PROCESSED']
APP_REPORTS = os.environ['APP_REPORTS']
APP_VERSION = os.environ['APP_VERSION']


def train_model(x_train, y_train, grid_params):

    ml_pipe = Pipeline([
        ('std_scaler', StandardScaler()),
        ('clf', tree.DecisionTreeClassifier())
    ])

    # Construct grid search
    gs = GridSearchCV(estimator=ml_pipe, scoring='roc_auc', n_jobs=-1,
                      param_grid=grid_params, cv=3)

    # Fit using grid search
    gs.fit(x_train, y_train)

    # Best accuracy
    print('Best training score: {0:.3f}'.format(gs.best_score_))

    # Best params
    print('\nBest params:\n', gs.best_params_)

    return gs.best_estimator_


def test_model(clf, x_train=None, x_test=None, y_train=None, y_test=None, saving=True):

    # Get a dummy score using the most frequent value
    most_frequent_value = y_train.value_counts().index[0]
    dummy_score = (most_frequent_value == y_test).mean()
    print(dummy_score, type(dummy_score), dummy_score.shape)

    real_score = clf.score(x_test, y_test)

    print('Testing score: {0:.3f}'.format(real_score))
    print('Dummy score: {0:.3f}'.format(dummy_score))
    if saving:
        print('Saving test score in reports/performance.csv....')
        # Reading and saving performance results
        try:
            performance = pd.read_csv(APP_REPORTS + 'performance.csv')
        except FileNotFoundError:
            performance = pd.DataFrame()
        performance.append(pd.DataFrame({'Classifier': ['Dummy (most frequent)', 'Decision Tree'],
                                         "Code Version": [APP_VERSION] * 2,
                                         'Type': ["Test"] * 2,
                                         'Score': [dummy_score, real_score],
                                         'Run': datetime.now()})) \
            .to_csv(APP_REPORTS + 'performance.csv', index=False)
