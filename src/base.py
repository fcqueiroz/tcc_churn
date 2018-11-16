# Standard Library Imports
import os
import pickle
from timeit import default_timer as timer
# Third Party Imports
import pandas as pd
# Local Application Imports
from src.data.make_dataset import table_cols_info as cols_info
from src.features.build_features import train_test_split
from src.models.train_model import train_model, test_model

# Global Variables
DATA_RAW = os.environ['DATA_RAW']
DATA_INTERIM = os.environ['DATA_INTERIM']
DATA_PROCESSED = os.environ['DATA_PROCESSED']
APP_MODELS = os.environ['APP_MODELS']
APP_REPORTS = os.environ['APP_REPORTS']
APP_VERSION = os.environ['APP_VERSION']


def _save_model(clf, filename=None):
    if filename is None:
        filename = APP_VERSION + '_decision_tree.sav'
    pickle.dump(clf, open(APP_MODELS + filename, 'wb'))


def _load_model(filename=None):
    if filename is None:
        filename = APP_VERSION + '_decision_tree.sav'
    clf = pickle.load(open(APP_MODELS + filename, 'rb'))

    return clf


def _split_raw():
    # Retrieve data from raw folder (assume it's already masked and cleaned data)
    raw_orders = pd.read_csv(DATA_RAW + 'orders.csv',
                             parse_dates=[cols_info['registry_date']],
                             index_col=0)

    # Get data in a proper format
    x_train, x_test, y_train, y_test = train_test_split(raw_orders)

    return x_train, x_test, y_train, y_test


def _save_processed(x_train, x_test, y_train, y_test):
    # Save train and test sets
    x_train.merge(y_train.to_frame(), left_index=True, right_index=True)\
        .to_csv(DATA_PROCESSED + 'train.csv')
    x_test.merge(y_test.to_frame(), left_index=True, right_index=True)\
        .to_csv(DATA_PROCESSED + 'test.csv')


def _load_processed():

    df_test = pd.read_csv(DATA_PROCESSED + 'test.csv', index_col=0)
    y_test = df_test['target']
    x_test = df_test.drop(columns='target')

    df_train = pd.read_csv(DATA_PROCESSED + 'train.csv', index_col=0)
    y_train = df_train['target']
    x_train = df_train.drop(columns='target')

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":

    model = None
    x_train = pd.DataFrame()
    x_test = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()

    # Set grid search params
    grid_params = [{
        'clf__criterion': ['gini', 'entropy'],
        'clf__class_weight': ['balanced'],
        'clf__random_state': [42]
        # 'clf__min_samples_leaf': np.linspace(0.001, 0.05, 5),
        # 'clf__max_depth': np.arange(1, 10, 1),
        # 'clf__min_samples_split': np.linspace(0.0001, 0.050, 5)
    }]

    print("Running script...\n")
    g_start = timer()
    batch = list(reversed(os.sys.argv[1:]))
    while True:
        # The script can run in batch mode if the options are passed as arguments of the call
        if len(batch) > 0:
            action = batch.pop()
        else:
            print("Options:",
                  "\n--------\n",
                  "train: Train model\n",
                  "test: Test model performance\n",
                  "load: Load trained model and processed datasets\n",
                  "save: Save trained model and processed datasets\n",
                  "exit: Exit program")
            action = input("What do you want to do? ").lower()

        if action == 'train':
            if x_train.empty or y_train.empty:
                print("Generating train and test set from raw data...")
                start = timer()
                x_train, x_test, y_train, y_test = _split_raw()
                end = timer()
                print("Finished. Train and test set loaded in memory. ({0:.1f} sec elapsed)".format(end - start))

            # Obtain best model
            print("Training model...")
            start = timer()
            model = train_model(x_train, y_train, grid_params)
            end = timer()
            print("Finished. The trained model was loaded in memory. ({0:.1f} sec elapsed)".format(end - start))
        elif action == 'load':
            print("Loading trained model and datasets...")
            start = timer()
            model = _load_model()
            x_train, x_test, y_train, y_test = _load_processed()
            end = timer()
            print("Finished. ({0:.1f} sec elapsed)".format(end - start))
        elif model is None and (action == 'test' or action == 'save'):
            print("There is no model loaded. Use 'train' or 'load' options.")
        elif action == 'test':
            print("Testing trained model...")
            start = timer()
            test_model(model, x_train, x_test, y_train, y_test)
            end = timer()
            print("Finished. ({0:.1f} sec elapsed)".format(end - start))
        elif action == 'save':
            print("Saving trained model and datasets...")
            start = timer()
            # Save trained model
            _save_model(model)
            # Save train and test sets in data/processed/
            _save_processed(x_train, x_test, y_train, y_test)
            end = timer()
            print("Finished. ({0:.1f} sec elapsed)".format(end - start))
        elif action == 'exit':
            g_end = timer()
            print("Total elapsed time: {0:.1f} seconds".format(g_end-g_start))
            break
        else:
            print("Unrecognized option '{}'".format(action))
