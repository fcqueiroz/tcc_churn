# Standard Library Imports
from datetime import datetime
import pickle
import os
# Third Party Imports
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# Global Variables
DATA_INTERIM = os.environ['DATA_INTERIM']
DATA_PROCESSED = os.environ['DATA_PROCESSED']
APP_MODELS = os.environ['APP_MODELS']
APP_REPORTS = os.environ['APP_REPORTS']
APP_VERSION = os.environ['APP_VERSION']


# TODO Turn this module into a class


def load_model(filename='decision_tree.sav'):
    filename = APP_MODELS + filename
    try:
        model = pickle.load(open(filename, 'rb'))
    except FileNotFoundError:
        print('The model was NOT found in {}'.format(filename))
        model = None
    return model


def save_model(model, filename='decision_tree.sav'):
    pickle.dump(model, open(APP_MODELS + filename, 'wb'))


def train_model(x_train, y_train, grid_params, scoring, cv=3):

    # Construct pipeline
    ml_pipe = Pipeline([
        ('clf', DecisionTreeClassifier())
    ])

    # Construct grid search
    gs = GridSearchCV(estimator=ml_pipe, scoring=scoring, param_grid=grid_params, n_jobs=-1, cv=cv)

    # Fit using grid search
    gs.fit(x_train, y_train)
    return gs


def test_model(model, x_test=None, y_test=None):

    # Get a dummy score using the most frequent value
    # most_frequent_value = y_train.value_counts().index[0]
    # dummy_pred = [most_frequent_value for v in range(0, len(y_test))]
    # Always predict positive class
    # dummy_pred = [True] * len(y_test)
    # dummy_score = f1_score(y_test, dummy_pred)

    real_score = model.score(x_test, y_test)

    print('\nBest params:\n', model.best_params_)
    # print('Dummy score: {0:.3f}'.format(dummy_score))
    print('Best training score: {0:.3f}'.format(model.best_score_))
    print('Testing score: {0:.3f}'.format(real_score))
    print('nodes: ', model.best_estimator_.named_steps['clf'].tree_.node_count)
    print('depth: ', model.best_estimator_.named_steps['clf'].tree_.max_depth, '\n')
    print(pd.Series(model.best_estimator_.named_steps['clf'].feature_importances_, index=x_test.columns)
          .sort_values(ascending=False).head(10), '\n')

    # Reading and saving performance results
    print('Saving test score in reports/performance.csv....')
    try:
        performance = pd.read_csv(APP_REPORTS + 'performance.csv')
    except FileNotFoundError:
        performance = pd.DataFrame()
    # TODO Include info about base churn rate, scoring method and training dataset size
    performance.append(pd.DataFrame({'Classifier': ['Decision Tree'],
                                     "Code Version": [APP_VERSION] ,
                                     'Type': ["Test"] ,
                                     'Score': [real_score],
                                     'Run': datetime.now()})) \
        .to_csv(APP_REPORTS + 'performance.csv', index=False)
