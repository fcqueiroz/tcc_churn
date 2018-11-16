# Standard Library Imports
from datetime import timedelta as timedelta
import os
# Third Party Imports
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
# Local Application Imports
from ..data.make_dataset import table_cols_info

# Global Variables
DATA_INTERIM = os.environ['DATA_INTERIM']


class OrderFeatures(BaseEstimator, TransformerMixin):
    """
    This class contains the methods to transform a raw dataframe in useful features
    for a machine learning algorithm.
    """

    registry_date_clm = table_cols_info['registry_date']
    registry_id_clm = table_cols_info['registry_id']
    registry_value_clm = table_cols_info['registry_value']
    customer_id_clm = table_cols_info['customer_id']

    def __init__(self, key, x_bound, y_bound):
        self.key = key
        self.x_date_bound = x_bound
        self.y_date_bound = y_bound

    def transform(self, dataframe, y=None):

        if self.key == 'recency':
            return self._build_recency(self._bounded_dataframe(dataframe))
        elif self.key == 'frequency':
            return self._build_frequency(self._bounded_dataframe(dataframe))
        elif self.key == 'target':
            return self._build_target(self._bounded_dataframe(dataframe, target=True))
        else: 
            pass

    def fit(self, x_data, y=None, **fit_params):
        return self

    def _bounded_dataframe(self, dataframe, target=False):
        if target:
            customer_bound = self._bounded_dataframe(dataframe)[self.customer_id_clm].unique()
            dataframe_bounded = dataframe.loc[
                (dataframe[self.registry_date_clm] < self.y_date_bound)
                & (dataframe[self.customer_id_clm].isin(customer_bound))
            ]
        else:
            dataframe_bounded = dataframe.loc[
                (dataframe[self.registry_date_clm] < self.x_date_bound)
            ]

        return dataframe_bounded

    def _build_recency(self, dataframe):
        """This feature is built as the age the customer had when they made their last purchase"""

        # This is first order's date
        fpo = dataframe.groupby(self.customer_id_clm)[self.registry_date_clm].min()
        # This is last order's date
        lpo = dataframe.groupby(self.customer_id_clm)[self.registry_date_clm].max()

        recency = (lpo - fpo).dt.days
        recency.name = 'recency'
        recency = recency.to_frame()
        return recency

    def _build_frequency(self, dataframe):
        """This feature is the number of time units (days, by default) where a recurring purchase was made"""

        # Group by the chosen time unit
        dataframe = dataframe[[self.customer_id_clm, self.registry_date_clm, self.registry_id_clm]]\
            .groupby([self.customer_id_clm, self.registry_date_clm]).count()\
            .reset_index()

        frequency = dataframe.groupby(self.customer_id_clm)[self.registry_id_clm].count() - 1
        frequency.name = 'frequency'
        frequency = frequency.to_frame()
        return frequency

    def _build_target(self, dataframe):
        """The target is defined as 'Is true that this customer won't place any order
        in the following 30 days?'
        """
        customer_last_order = dataframe.groupby(self.customer_id_clm)[self.registry_date_clm].max()
        last_registry_date = dataframe[self.registry_date_clm].max()
        customer_days_wo_orders = pd.to_timedelta(last_registry_date - customer_last_order)
        target = customer_days_wo_orders >= timedelta(days=30)
            
        target = target.to_frame()
        target.name = 'target'
        return target


def temporal_split(date_range, splits=2, subset_interval=14, censored_interval=30, min_period=30, time_unit='d'):
    """
    This method splits the date period in subsets for two reasons:
    1) Correctly calculate the target. Since the churn event can only be acknowledged as
    a fact only 30 days after it happened , we need to created a censored period between
    the last available input and the date when the target is assessed.
    2) Prepare for including nested cross-validation and walk forward optimization in
    the time series.

    Parameters
    ----------
    date_range : datetime array
        Array containing the possible values for the data being used

    splits : int
        Number of subsets containing input and output values. Default option generates
        2 subsets, one for training and other for testing the model.

    subset_interval : int
        Number of time units between consecutive subsets

    censored_interval : int
        Number of time units between the churn event and the acknowledgement of this event

    min_period : int
        Minimum number of time units a subset may contain.

    time_unit : str
        Offset alias for determing the time frequency and correctly converting the inputs to timedeltas
    """

    if time_unit == 'd':
        censored_timedelta = timedelta(days=censored_interval)
        subset_timedelta = timedelta(days=subset_interval)
        min_period_timedelta = timedelta(days=min_period)
    elif time_unit == 'w':
        censored_timedelta = timedelta(weeks=censored_interval)
        subset_timedelta = timedelta(weeks=subset_interval)
        min_period_timedelta = timedelta(weeks=min_period)
    else:
        pass

    if splits == 'max':
        splits = (date_range.max() - date_range.min() - min_period_timedelta) // subset_timedelta + 1

    y_max_date_splits = [date_range.max() - k*subset_timedelta for k in range(0, splits)]
    x_max_date_splits = [i - censored_timedelta for i in y_max_date_splits]

    return y_max_date_splits, x_max_date_splits


def wrangling_pipe(raw_data, x_bound, y_bound):
    """
    A pipeline for producing x_data based on table that isn't in a suitable
    shape for a machine learning process

    Parameters
    ----------
    raw_data : pandas.DataFrame object
        Pandas dataframe object to be transformed in a dataset suitable for
        machine learning processing.

    x_bound : timestamp
        Date that represents the upper limit of the visible period for producing the X table.
        In other words, only things that happened before the x_bound date are available
        for producing features for the X table.

    y_bound : timestamp
        Date that represents the upper limit of the visible period for producing the y array.
        In other words, only things that happened before the y_bound date are available
        for calculating the target.

    Returns
    -------
    x_data : Pandas Dataframe object of shape [n_samples, n_features]
        Table containing the input variables.

    y_data: Pandas Dataframe object of shape [n_samples]
        Array containing the output variables.
    """
    # Build a pipe for each feature that will be extracted from the raw data
    recency_pipe = Pipeline([
        ('build_recency', OrderFeatures('recency', x_bound, y_bound))
    ])
    frequency_pipe = Pipeline([
        ('build_frequency', OrderFeatures('frequency', x_bound, y_bound))
    ])
    target_pipe = Pipeline([
        ('build_target', OrderFeatures('target', x_bound, y_bound))
    ])

    # Feature union
    union_pipe = FeatureUnion([
        ('recency', recency_pipe),
        ('frequency', frequency_pipe),
        ('target', target_pipe)
    ])

    # Transform visible raw data to a format useful for the machine learning part
    df = pd.DataFrame(
        union_pipe.fit_transform(raw_data),
        columns=['recency', 'frequency', 'target']
    )

    y_data = df['target']
    x_data = df.drop(columns='target')

    return x_data, y_data


def train_test_split(raw_data):
    """
    Produces a split in train and test datasets, considering that we are dealing with
    a time series.

    Parameters
    ----------
    raw_data : pandas.DataFrame object
        Pandas dataframe object to be transformed in two datasets suitables for
        machine learning processing.

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

    # Floor dataframe date
    date_range = raw_data[table_cols_info['registry_date']].dt.floor('d')
    raw_data[table_cols_info['registry_date']] = date_range

    # Generate idx for X,y,train,test with temporal_split()
    y_bound, x_bound = temporal_split(date_range)

    # Transforms raw data for making it more appropriate for machine learning purposes
    x_test, y_test = wrangling_pipe(raw_data, x_bound[0], y_bound[0])
    x_train, y_train = wrangling_pipe(raw_data, x_bound[1], y_bound[1])

    return x_train, x_test, y_train, y_test
