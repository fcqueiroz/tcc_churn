import os
import yaml
import pandas as pd
from datetime import datetime
import pickle

script_dir = os.path.dirname(__file__)

# Get global settings file (including information about the path to the data folder)
rel_path = "../../config.yml"
abs_file_path = os.path.join(script_dir, rel_path)
with open(abs_file_path, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)
    paths = cfg['paths']
    app = cfg['app']


if __name__ == "__main__":

    df_train = pd.read_csv(paths['processed'] + 'train.csv', index_col=0)
    df_test = pd.read_csv(paths['processed'] + 'test.csv', index_col=0)
    y_test = df_test[['target']]
    X_test = df_test.drop(columns='target')

    # Get a dummy score using the most frequent value
    most_frequent_value = df_train['target'].value_counts().index[0]
    dummy_score = (most_frequent_value == y_test)['target'].values.mean()

    # Load & run the real model
    filename = app['version'] + '_decision_tree.sav'
    clf = pickle.load(open(paths['models'] + filename, 'rb'))
    real_score = clf.score(X_test, y_test)

    # Reading and saving performance results
    try:
        performance = pd.read_csv(paths['reports'] + 'performance.csv')
    except FileNotFoundError:
        performance = pd.DataFrame()
    performance.append(pd.DataFrame({'Classifier': ['Dummy (most frequent)', 'Decision Tree (unoptimized)'],
                                     "Code Version": [app['version']] * 2,
                                     'Type': ["Test"] * 2,
                                     'Score': [dummy_score, real_score],
                                     'Run': datetime.now()})) \
        .to_csv(paths['reports'] + 'performance.csv', index=False)
