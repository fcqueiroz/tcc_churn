import os
import yaml
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
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

    y_train = df_train[['target']]
    X_train = df_train.drop(columns='target')

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    filename = app['version'] + '_decision_tree.sav'
    pickle.dump(clf, open(paths['models'] + filename, 'wb'))
