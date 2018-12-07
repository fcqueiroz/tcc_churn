####################################
# Script to download/generate data #
####################################

"""
I can't provide any data yet, but if you can generate your own
(and save in data/raw/orders.csv) then all the provided code
should be ready to work for you!
The dataset being used is a orders table containing for each row:
registry_id: The id of the order
registry_date: The date the order was created
registry_value: The value of that order
customer_id: The id of the customer who placed the order
registry_status: The status of a certain order
"""

# Write here the name of the columns being used for each of these info

table_cols_info = {
    'registry_date': 'date_column_name',
    'registry_id': 'order_id_column_name',
    'registry_value': 'order_value_column_name',
    'customer_id': 'customer_id_column_name',
    'registry_status': 'order_status'
}
