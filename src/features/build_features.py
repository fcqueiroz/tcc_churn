# Standard Library Imports
import csv
from datetime import timedelta as timedelta
import os
# Third Party Imports
import pandas as pd
import pandas_profiling
import sqlite3
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from tsfresh import extract_features as ts_extract_feat
from tsfresh.feature_extraction import settings
from tsfresh.utilities.dataframe_functions import impute
# Local Application Imports
from ..data.make_dataset import queries, table_cols_info as orders_cols

# Global Variables
DATA_INTERIM = os.environ['DATA_INTERIM']
DATA_PROCESSED = os.environ['DATA_PROCESSED']
DATA_RAW = os.environ['DATA_RAW']
APP_REPORTS = os.environ['APP_REPORTS']


class BaseFeatures(BaseEstimator, TransformerMixin):
    """Abstract base class for database feature transformers"""

    registry_date_clm = orders_cols['registry_date']
    registry_id_clm = orders_cols['registry_id']
    registry_val_clm = orders_cols['registry_value']
    customer_id_clm = orders_cols['customer_id']

    def __init__(self, key):
        self.key = key
        self.names = None

    def fit(self, x_data, y=None, **fit_params):
        return self

    def transform(self, dataframe, y=None):
        build_method = '_build_' + self.key
        feature = getattr(self, build_method)(dataframe)
        return feature

    def get_feature_names(self):
        return self.names


class OrderFeatures(BaseFeatures):
    """
    This class contains the methods to transform the orders raw dataframe in useful features
    for a machine learning algorithm.
    """

    # Target censored window in days
    target_cw = 30
    try:
        # This file contains a list of the filtered relevant features
        with open(DATA_INTERIM + 'tsfresh_train_filtered.csv', 'r') as fin:
            csvin = csv.reader(fin)
            tsfresh_relevant_columns = next(csvin)
    except FileNotFoundError:
        tsfresh_relevant_columns = []

    def __init__(self, feature, offset):
        self.offset = offset
        super().__init__(feature)

    def transform(self, dataframe, y=None):
        # Change dataframe to include offset
        dataframe = self._bounded_dataframe(dataframe)
        feature = super().transform(dataframe, y)
        return feature

    def _bounded_dataframe(self, df):
        # Find which customers belong to input subset
        customer_input = df.loc[
            df[self.registry_date_clm] < (
                df[self.registry_date_clm].max() - timedelta(days=(self.offset + self.target_cw))),
            self.customer_id_clm].unique()

        if self.key == 'target':
            # If the target is being created, remove the offset and keep only the clients that belong to the input
            df = df.loc[(df[self.registry_date_clm] < (df[self.registry_date_clm].max()
                                                       - timedelta(days=self.offset))) &
                        (df[self.customer_id_clm].isin(customer_input))]
        else:
            # If something other than the target is being created, remove the offset plus the target censored period
            df = df.loc[df[self.registry_date_clm] < (df[self.registry_date_clm].max()
                                                      - timedelta(days=(self.offset + self.target_cw)))]
        return df

    def _build_tsfresh(self, dataframe):
        if not self.tsfresh_relevant_columns:
            # If a list of filtered relevant features is provided, generate only those
            features = ts_extract_feat(dataframe, impute_function=impute,
                                       column_id=self.customer_id_clm, column_sort=self.registry_date_clm,
                                       kind_to_fc_parameters=settings.from_columns(self.tsfresh_relevant_columns))
        else:
            # Otherwise, generate all the features (this might take a WHILE depending on the dataset size)
            features = ts_extract_feat(dataframe, impute_function=impute,
                                       column_id=self.customer_id_clm, column_sort=self.registry_date_clm)
        self.names = list(features.columns)
        return features

    def _build_recency_v0(self, dataframe):
        """This feature is built as the age the customer had when they made their last purchase"""

        # This is first order's date
        fpo = dataframe.groupby(self.customer_id_clm)[self.registry_date_clm].min()
        # This is last order's date
        lpo = dataframe.groupby(self.customer_id_clm)[self.registry_date_clm].max()

        feature = (lpo - fpo).dt.days + 1
        feature.name = self.key
        self.names = [self.key]
        feature = feature.to_frame()
        return feature

    def _build_recency_v1(self, dataframe):
        """This feature represents the number of days that passed since the customer last order"""

        customer_last_order = dataframe.groupby(self.customer_id_clm)[self.registry_date_clm].max()
        last_registry_date = dataframe[self.registry_date_clm].max()
        # Days without making any order, for each customer
        feature = (last_registry_date - customer_last_order).dt.days
        feature.name = self.key
        self.names = [self.key]
        feature = feature.to_frame()
        return feature

    def _build_c_age(self, dataframe):
        customer_first_order = dataframe.groupby(self.customer_id_clm)[self.registry_date_clm].min()

        last_registry_date = dataframe[self.registry_date_clm].max()
        feature = (last_registry_date - customer_first_order).dt.days + 1
        self.names = [self.key]
        feature.name = self.key
        feature = feature.to_frame()
        return feature

    def _build_gravity(self, dataframe):
        customer_first_order = dataframe.groupby(self.customer_id_clm)[self.registry_date_clm].min()
        customer_first_order.name = 'fpo'
        df = dataframe[
            [self.customer_id_clm, self.registry_date_clm]
        ].merge(customer_first_order.to_frame(),
                left_on=self.customer_id_clm, right_index=True).set_index(self.customer_id_clm)
        df['gravity'] = (df.loc[:, self.registry_date_clm] - df.loc[:, 'fpo']).dt.days + 1
        w = df.groupby(self.customer_id_clm)['gravity'].sum() / df.groupby(self.customer_id_clm)['gravity'].count()
        w = w.to_frame()
        # Normalize gravity between 0.0 and 1.0
        w['gravity_norm'] = (w / self._build_c_age(dataframe)).iloc[:, 0]
        self.names = list(w.columns)
        return w

    def _build_activity(self, dataframe):
        pass  # qtd & % dos periodos em que esteve ativo

    def _build_bg_nbd(self, dataframe):
        pass  # Calculate the probability for the client being alive in the present

    def _build_monetary(self, dataframe):
        feature = dataframe.groupby(self.customer_id_clm)[self.registry_val_clm].sum()
        self.names = [self.key]
        feature.name = self.key
        feature = feature.to_frame()
        return feature

    def _build_frequency(self, dataframe):
        """This feature is the number of time units (days, by default) where a recurring purchase was made"""
        # Group by the chosen time unit
        df = dataframe[[self.customer_id_clm, self.registry_date_clm]].reset_index()\
            .groupby([self.customer_id_clm, self.registry_date_clm]).count()\
            .reset_index()
        feature = df.groupby(self.customer_id_clm)[self.registry_id_clm].count() - 1
        feature.name = self.key
        feature = feature.to_frame()
        # Normalize feature between 0.0 and 1.0
        feature['frequency_norm'] = (feature / self._build_c_age(dataframe)).iloc[:, 0]
        self.names = list(feature.columns)
        return feature

    def _build_target(self, dataframe):
        """The target is defined as positive (churn) when an alive customer (that placed an order in
        the last 30 days) will be dead by the end of the next 30 days period (won't place any order
        in the following 30 days)
        """
        # worders: Days without making any order, for each customer
        worders = self._build_recency_v1(dataframe)
        feature = (worders >= self.target_cw) & (worders < (2*self.target_cw))

        if len(feature.iloc[:, 0].unique()) == 1:
            raise ValueError("The target has ONE unique value")
        self.names = [self.key]
        return feature

    def _build_index(self, dataframe):
        feature = dataframe.groupby(self.customer_id_clm).count().index
        self.names = [self.key]
        feature.name = self.key
        feature = feature.to_frame()
        return feature


def feat_pipe(feats, offset):
    """Build a pipe for each feature that will be extracted from the raw data"""
    for f in feats:
        yield (f, OrderFeatures(f, offset))


def wrangling_pipe(raw_data, feats, offset=0):
    """
    A pipeline for producing x_data based on table that isn't in a suitable
    shape for a machine learning process

    Parameters
    ----------
    raw_data : pandas.DataFrame object
        Pandas dataframe object to be transformed in a dataset suitable for
        machine learning processing.

    feats : list
        Features to be built

    offset : int
        Number of days ahead the test subset is from the train subset

    Returns
    -------
    x_data : Pandas Dataframe object of shape [n_samples, n_features]
        Array containing the input set.

    y_data : Pandas Series object of shape [n_samples]
        Array containing the output set.
    """

    # Unite all features
    *pipe, = feat_pipe(feats, offset)
    union_pipe = FeatureUnion(pipe)

    # Transform visible raw data to a format useful for the machine learning part
    result = union_pipe.fit_transform(raw_data)
    df = pd.DataFrame(result, columns=union_pipe.get_feature_names())
    # Rename target and index
    df.rename(inplace=True, columns={'target__target': 'target', 'index__index': 'client'})

    df.set_index(inplace=True, keys='client')

    # Drop duplicate rows
    df = df.loc[~df.duplicated()]

    y_data = df['target']
    x_data = df.drop(columns='target')

    return x_data, y_data


def transform_raw(feats, train_test_delta=60, report_warning=100, random_state=42, n=None, frac=None):
    """
    Transforms raw data into train and test datasets, considering that we are dealing with
    a time series.

    Parameters
    ----------
    feats : list
        Features to be built

    train_test_delta : int
        How many days ahead the test subset is from the train subset

    report_warning : int
        safety value to prevent running a profiling on a very big dataset. The pandas_profiling
        will execute only if the number of columns is smaller than report_warning.

    n : int, optional
        Number of items from axis to return. Cannot be used with `frac`.
        Default = 1 if `frac` = None.

    frac : float, optional
        Fraction of axis items to return. Cannot be used with `n`.

    random_state : int or numpy.random.RandomState, optional
        Seed for the random number generator (if int), or numpy RandomState object.

    Returns
    -------
    x_train : Pandas Dataframe object of shape [n_samples, n_features]
        Training set.

    y_train : Pandas Dataframe object of shape [n_samples]
        Target values for training set.

    x_test : Pandas Dataframe object of shape [n_samples, n_features]
        Testing set.

    y_test : Pandas Dataframe object of shape [n_samples]
        Target values for testing set.
    """
    # Open Database connection
    con = sqlite3.connect(DATA_RAW + 'raw.db', check_same_thread=False)

    # Obtain a sampled and shuffled list of customers
    clients = pd.read_sql_query(queries['customers'], con, index_col=orders_cols['customer_id'])\
        .sample(frac=frac, random_state=random_state, n=n)\
        .index
    client_half_len = int(len(clients)/2)

    # Retrieve data from raw folder (assume it's already masked and cleaned data)
    raw_orders = pd.read_sql_query(queries['raw_df'].format(tuple([i for i in clients])), con,
                                   index_col=orders_cols['registry_id'], parse_dates=[orders_cols['registry_date']])
    # Close database connection
    con.close()

    # Floor dataframe date
    date_range = raw_orders[orders_cols['registry_date']].dt.floor('d')
    raw_orders[orders_cols['registry_date']] = date_range

    # Pick a subset of clients for training
    selected_clients_raw_orders = raw_orders.loc[raw_orders[orders_cols['customer_id']].isin(clients[:client_half_len])]
    # Transforms raw data for making it more appropriate for machine learning purposes
    x_train, y_train = wrangling_pipe(selected_clients_raw_orders, feats, train_test_delta)

    # TODO prevent the test set to generate features that are out of training subset
    selected_clients_raw_orders = raw_orders.loc[raw_orders[orders_cols['customer_id']].isin(clients[client_half_len:])]
    x_test, y_test = wrangling_pipe(selected_clients_raw_orders, feats)

    # Analyze the dataset
    if x_train.shape[1] > report_warning:
        print('\nThe dataset is too big. It will NOT be analyzed.')
    else:
        profile = pandas_profiling.ProfileReport(x_train)
        profile.to_file(APP_REPORTS + '/profiling.html')
        # Get list of high correlated columns
        rejected = profile.get_rejected_variables(threshold=0.9)
        # TODO Drop highly correlated columns before running a pandas_profiling
        # Drop columns with big correlation
        x_train.drop(inplace=True, columns=rejected)
        x_test.drop(inplace=True, columns=rejected)

    # TODO test that the train and test contain different data in at least X%

    return x_train, x_test, y_train, y_test


def save_processed(x_data, y_data, name):
    """
    Save processed datasets in disk

    Parameters
    ----------
    x_data : Pandas Dataframe object of shape [n_samples, n_features]
        Array containing the input set.

    y_data : Pandas Series object of shape [n_samples]
        Array containing the output set.

    name : str
        The name for the saved file, without extension (it's implied .csv)
    """
    x_data.merge(y_data.to_frame(), left_index=True, right_index=True)\
        .to_csv(DATA_PROCESSED + name + '.csv')


def load_processed(name):
    """
    Load processed datasets from disk

    Parameters
    ----------
    name : str
        The name for the saved file, without extension (it's implied .csv)

    Returns
    -------
    x_data : Pandas Dataframe object of shape [n_samples, n_features]
        Array containing the input set.

    y_data : Pandas Series object of shape [n_samples]
        Array containing the output set.
    """
    df = pd.read_csv(DATA_PROCESSED + name + '.csv', index_col=0)
    y_data = df['target']
    x_data = df.drop(columns='target')

    return x_data, y_data
