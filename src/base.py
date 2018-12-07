# Standard Library Imports
import os
from timeit import default_timer as timer
# Third Party Imports
import numpy as np
import pandas as pd
# Local Application Imports
from src.data.make_dataset import make_dataset
from src.features.build_features import transform_raw, load_processed, save_processed
from src.models.train_model import train_model, test_model, load_model, save_model
from src.visualization.visualize import plot_learning_curve, plot_confusion_matrix, plot_roc_curve, plot_tree
from src.visualization.visualize import plot_precision_recall_curve


if __name__ == "__main__":

    table_names = ['orders']
    model = None
    x_train = pd.DataFrame()
    x_test = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()

    # Maximum number of nodes allowed for exporting the tree structure
    nodes_warning = 100
    # Maximum number of columns allowed to generate a pandas_profiling
    report_warning = 100

    # Features to be built
    feats = ['index', 'recency_v0', 'recency_v1', 'c_age', 'gravity', 'monetary', 'frequency', 'tsfresh', 'target']
    scoring = 'roc_auc'

    # Number of cross-validations folds
    cv = 5
    # Prints title in figures.
    title = False

    # Set grid search params
    grid_params = [{
        # 'clf__criterion': ['gini', 'entropy'],
        # 'clf__class_weight': ['balanced', None],
        'clf__min_samples_split': np.arange(50, 200, 20),
        'clf__min_impurity_decrease': np.linspace(0.0, 0.002, 5),
        'clf__max_depth': np.arange(3, 6, 1),
        'clf__random_state': [42]
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
                  "get: Make Dataset\n",
                  "train: Train model\n",
                  "transform: Transfrom raw data in features using tsfresh\n",
                  "test: Test model performance\n",
                  "load: Load trained model and processed datasets\n",
                  "save: Save trained model and processed datasets\n",
                  "exit: Exit program")
            action = input("What do you want to do? ").lower()

        if action == 'get':
            print("Obtaining raw data from servers...")
            start = timer()
            for t in table_names:
                make_dataset(t)
                print("Table '{}' was downloaded".format(t))
            end = timer()
            print("Finished. ({0:.1f} sec elapsed)\n".format(end - start))
        elif action == 'train':
            if x_train.empty or y_train.empty:
                print("Generating train and test set from raw data...")
                start = timer()
                x_train, x_test, y_train, y_test = transform_raw(feats, report_warning=report_warning, frac=0.99)
                # TODO Test that the returned types are dataframes for X and series for Y.
                # TODO Test the shapes are correct
                # TODO Test the Y has different values
                end = timer()
                print("Finished. Train and test set loaded in memory. ({0:.1f} sec elapsed)\n".format(end - start))

            # Obtain best model
            print("Training model...")
            start = timer()
            model = train_model(x_train, y_train, grid_params, scoring, cv=cv)
            end = timer()
            print("Finished. The trained model was loaded in memory. ({0:.1f} sec elapsed)\n".format(end - start))
        elif action == 'transform':
            print("Generating train and test set from raw data...")
            start = timer()
            x_train, x_test, y_train, y_test = transform_raw(feats, report_warning=report_warning, frac=0.20)
            end = timer()
            print("Finished. Train and test set loaded in memory. ({0:.1f} sec elapsed)\n".format(end - start))
        elif action == 'load':
            print("Loading trained model and datasets...")
            start = timer()
            model = load_model()
            x_train, y_train = load_processed(name='train')
            x_test, y_test = load_processed(name='test')
            # TODO Test that the returned types are dataframes for X and series for Y.
            # TODO Test the shapes are correct
            # TODO Test the Y has different values
            end = timer()
            print("Finished. ({0:.1f} sec elapsed)\n".format(end - start))
        elif action == 'test':
            if model is None:
                print("There is no model loaded. Use 'train' or 'load' options.")
            else:
                print("Testing trained model...")
                start = timer()
                test_model(model, x_test, y_test)
                end = timer()
                print("Finished. ({0:.1f} sec elapsed)\n".format(end - start))
        elif action == 'save':
            if x_train.empty or y_train.empty or x_test.empty or y_test.empty:
                print("There is nothing to save. Use 'train' or 'load' options.")
            else:
                start = timer()
                if model is None:
                    print("No model detected. Saving only datasets...")
                else:
                    print("Saving trained model and datasets...")
                    save_model(model)
                    if model.best_estimator_.named_steps['clf'].tree_.node_count > nodes_warning:
                        print('The tree is too big. The structure WAS NOT exported to a figure')
                    else:
                        print('Saving decision tree in reports/tree.png....')
                        plot_tree(model.best_estimator_.named_steps['clf'], x_train.columns)

                    print('Saving confusion matrix in reports/...')
                    plot_confusion_matrix(model.best_estimator_, x_train, y_train, title=title)
                    print('Saving plotted curves in reports/...')
                    plot_precision_recall_curve(model.best_estimator_, x_train, y_train, title=title)
                    plot_roc_curve(model.best_estimator_, x_train, y_train, x_test, y_test, title=title)
                    plot_learning_curve(model.best_estimator_, x_train, y_train, scoring, cv=cv, title=title)
                # Save train and test sets in data/processed/
                save_processed(x_train, y_train, name='train')
                save_processed(x_test, y_test, name='test')
                end = timer()
                print("Finished. ({0:.1f} sec elapsed)\n".format(end - start))
        elif action == 'exit':
            g_end = timer()
            print("Total elapsed time: {0:.1f} seconds\n".format(g_end-g_start))
            break
        else:
            print("Unrecognized option '{}'".format(action))
