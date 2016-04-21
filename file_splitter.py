import csv
import pandas as pd
import numpy as np


def parse_dates(df):
    search_date = list(df['date_time'])
    chkin_date = list(df['srch_ci'])
    chkout_date = list(df['srch_co'])
    return df

"""
Variables
"""
# Sampling rate of the data
samp = 100
# Whether to merge the data
merge = False
# sample_train filename, None if not required
train_file = 'input/train_samp_%d_merged.csv' % samp
"""
End of variables
"""
# Read destinations
destinations = pd.DataFrame.from_csv('input/destinations.csv')
# print(destinations)

# Read and sample train
with open('input/train.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    columns = spamreader.__next__()
    print('Read the train table columns')
    print(len(columns))
    train_samp = []
    for i, row in enumerate(spamreader):
        if i % samp == 0:
            train_samp.append(row)
        if i % 1000000 == 0:
            print(i)
csvfile.close()

# build pandas Dataframe
train_samp = pd.DataFrame(train_samp, columns=columns)
train_samp.index = train_samp.user_id
train_samp.hotel_cluster = train_samp.hotel_cluster.astype(int)

# Merge if required
if merge:
    train_samp = pd.merge(train_samp, destinations, left_on=train_samp.srch_destination_id.values.astype(int),
                          right_on=destinations.index.values, how='left')

# Export train file
if train_file:
    train_samp.to_csv(train_file)

# Separate X_train and y_train
target = train_samp.hotel_cluster

# Removing excess columns in train
del train_samp['hotel_cluster']
del train_samp['cnt']
del train_samp['is_booking']
print(train_samp)
# print(target)

# Read test
test = pd.read_csv('input/test.csv', index_col=0)
print(test)
# Removing excess columns in test

print(sorted(train_samp.columns.values))
print(sorted(test.columns.values))
