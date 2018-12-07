# Standard Library Imports
import os
# Third Party Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydotplus
import seaborn as sns
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import learning_curve, validation_curve

# Global Variables
APP_REPORTS = os.environ['APP_REPORTS']


def plot_confusion_matrix(model, X, y, figsize=(4.2, 3.4), title=True):
    # Confusion Matrix calculations
    cm = pd.DataFrame(confusion_matrix(y, model.predict(X)), columns=['No', 'Yes'], index=['No', 'Yes'])
    cm = cm.applymap(lambda x: x / len(y))

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    g1 = sns.heatmap(cm, annot=True)
    if title:
        g1.set_title('Confusion Matrix [% of population]')
    g1.set_ylabel('Real Value')
    g1.set_xlabel('Predicted Value')
    plt.tight_layout()
    fig.savefig(APP_REPORTS + 'confusion_matrix.png')


def plot_roc_curve(model, x_train, y_train, x_test=None, y_test=None, figsize=(4.2, 4), title=True):
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Calculating and Plotting Dummy Estimator
    pred = [y_train.value_counts().index[0]] * len(y_train)
    fpr, tpr, thresholds = roc_curve(y_train, pred)
    ax.plot(fpr, tpr, label='Random Classifier')

    # Calculating and Plotting Real Estimator
    pred = model.predict_proba(x_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, pred)
    ax.plot(fpr, tpr, '--', color="#111111",
            label='Training Data AUC Score: {:0.2f}'.format(roc_auc_score(y_train, pred)))

    if x_test is not None and y_test is not None:
        # Calculating and Plotting Real Estimator
        pred = model.predict_proba(x_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, pred)
        ax.plot(fpr, tpr, color="#111111",
                label='Testing Data AUC Score: {:0.2f}'.format(roc_auc_score(y_test, pred)))

    # Adjusting and saving figure
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    if title:
        ax.set_title("ROC Curve for Decision Tree Classifier")
    ax.legend()
    plt.tight_layout()
    fig.savefig(APP_REPORTS + 'roc_curve.png')


def plot_learning_curve(model, x_train, y_train, scoring, x_test=None, y_test=None, cv=3, random_state=42, n_jobs=-1,
                        shuffle=True, train_sizes=np.linspace(0.02, 0.50, 30), figsize=(7, 3.8), title=True, *args):

    # TODO Use real testing data for plotting learning curve
    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(model, x_train, y_train, random_state=random_state,
                                                            scoring=scoring, n_jobs=n_jobs, cv=cv, shuffle=shuffle,
                                                            train_sizes=train_sizes, *args)

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Draw lines
    ax.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    ax.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    # Draw bands
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD", alpha=0.7)
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD", alpha=0.7)

    if title:
        plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Score"), plt.legend(loc="best")
    plt.tight_layout()
    fig.savefig(APP_REPORTS + 'learning_curve.png')


def plot_tree(clf, names):
    dot_data = StringIO()
    export_graphviz(clf, feature_names=names, class_names=['other', 'churn'],
                    out_file=dot_data, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    filename = APP_REPORTS + 'tree.png'
    graph.write_png(filename)


def plot_precision_recall_curve(model, X, y, figsize=(4, 4), title=True):
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Model's calculations
    pred = model.predict(X)
    precision, recall, _ = precision_recall_curve(y, pred)

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    if title:
        plt.title('Binary-class Precision-Recall curve')
    plt.tight_layout()
    fig.savefig(APP_REPORTS + 'precision_recall_curve.png')


def plot_validation_curve(model, X, y, scoring, param_name, param_range, cv=3, figsize=(4, 4), title=True):
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Model's calculations
    train_scores, valid_scores = validation_curve(model, X, y, param_name, param_range, cv=cv,
                                                  scoring=scoring, n_jobs=-1)

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(valid_scores, axis=1)
    test_std = np.std(valid_scores, axis=1)

    # Draw lines
    ax.plot(param_range, train_mean, '--', color="#111111", label='Training score')
    ax.plot(param_range, test_mean, color="#111111", label='Cross-validation score')

    # Draw bands
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="#DDDDDD", alpha=0.7)
    ax.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="#DDDDDD", alpha=0.7)

    if title:
        plt.title("Validation Curve")
    plt.xlabel(param_name), plt.ylabel("Score"), plt.legend(loc="best")
    plt.tight_layout()
    fig.savefig(APP_REPORTS + 'validation_curve_' + param_name + '.png')
