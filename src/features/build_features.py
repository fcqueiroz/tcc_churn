# imports
import yaml
import os
import pandas as pd
from datetime import timedelta as timedelta

script_dir = os.path.dirname(__file__)

# Get global settings file (including information about the path to the data folder)
rel_path = "../../config.yml"
abs_file_path = os.path.join(script_dir, rel_path)
with open(abs_file_path, 'r') as ymlfile:
    paths = yaml.load(ymlfile)['paths']

class Features(object):

    # Read dataframe
    dataframe = pd.read_csv(paths['interim'] + 'orders.csv', parse_dates=['created'])

    def __init__(self, name, test_offset=None):
        self.name = name
        self.present = self.get_present_time(test_offset)
        self.feature = None

        self.build()
        #self.export()

    def get_present_time(self, test_offset):
        return self.dataframe.created.dt.floor('d').max() - test_offset

    def export(self):
        self.feature.to_csv(paths['interim']+'ft_'+self.name+'.csv')

    def build(self):
        if self.name == 'recency':
            self.build_recency()
        elif self.name == 'frequency':
            self.build_frequency()
        elif self.name == 'target':
            self.build_target()
        else:
            pass

    def build_recency(self):
        # Filtering only the useful data
        condition = (self.dataframe.received > 0) & (self.dataframe.created < self.present)
        selected_cols = ['customer_db_id', 'created']
        df = self.dataframe.loc[condition, selected_cols]

        # This is first paid order date
        fpo = df.groupby('customer_db_id').created.min().dt.floor('d')
        fpo.name = 'first_paid_order'

        # This is last paid order date
        lpo = df.groupby('customer_db_id').created.max().dt.floor('d')
        lpo.name = 'last_paid_order'

        df = pd.DataFrame({'first_paid_order': fpo, 'last_paid_order': lpo})

        # This feature is built as the age the customer had when they made their last purchase
        self.feature = (df.last_paid_order - df.first_paid_order).dt.days
        self.feature.name = self.name

    def build_frequency(self):
        # Filtering only the useful data
        condition = (self.dataframe.received > 0) & (self.dataframe.created < self.present)
        selected_cols = ['customer_db_id', 'created', 'db_id']
        df = self.dataframe.loc[condition, selected_cols]

        # Group by the chosen time unit
        df = df.groupby(['customer_db_id', 'created']).count().reset_index()

        # This feature is the number of days where a REPEAT purchase was made
        self.feature = df.groupby('customer_db_id').count().db_id - 1
        self.feature.name = self.name

    def build_target(self):
        # Filtering only the useful data
        condition = (self.dataframe.received > 0) & (self.dataframe.created < self.present)
        selected_cols = ['customer_db_id', 'created']
        df = self.dataframe.loc[condition, selected_cols]

        # Current date
        today = df.created.dt.floor('d').max()

        # Last purchase
        lpo = df.groupby('customer_db_id').created.max().dt.floor('d')

        # The target is simply 'yes' if the customer made their last purchase more than 30 days ago
        self.feature = (today - lpo).dt.days > 30
        self.feature.name = self.name


def merge_features(features_list, test_offset=None):

    # Generate feature
    feature_name = features_list[0]
    print("Building {} feature".format(feature_name))
    dataframe = Features(feature_name, test_offset).feature.reset_index()

    # Merge
    for n in range(1, len(features_list)):
        feature_name = features_list[n]
        print("Building {} feature".format(feature_name))
        dataframe = dataframe.merge(Features(feature_name, test_offset).feature.reset_index())

    dataframe.set_index('customer_db_id', inplace=True)
    return dataframe


if __name__ == "__main__":

    """This method applies a filter on the original database to separate the historical
    data into past and future information. This is made by picking an arbitrary data in
    past and considering that as the 'present' moment. This way, we can use historical
    data to test future behavior.
    """

    # Construction parameters
    features = ['recency', 'frequency', 'target']
    # The test period is the period of most recent data that will be
    # separated from the model to emulate future (unseen) data
    test_period = timedelta(days=30)

    # Generate training data
    print("\nGenerating train dataset...")
    mf = merge_features(features, test_period)
    mf.to_csv(paths['processed']+'train'+'.csv')

    # Generate test data
    print("\nGenerating test dataset...")
    mf = merge_features(features, timedelta(days=0))
    mf.to_csv(paths['processed'] + 'test' + '.csv')
