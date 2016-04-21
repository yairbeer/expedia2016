import csv
import pandas as pd
import numpy as np
import datetime
from ml_metrics import mapk
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split


def parse_dates(df):
    search_date = list(df['date_time'])
    search_date = list(map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'), search_date))
    df['search_weekday'] = list(map(lambda x: int(x.strftime('%w')), search_date))
    df['search_month'] = list(map(lambda x: int(x.strftime('%m')), search_date))
    df['search_monthday'] = list(map(lambda x: int(x.strftime('%d')), search_date))
    del df['date_time']
    chkin_date = list(df['srch_ci'])
    chkin_date = list(map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'), chkin_date))
    df['ci_weekday'] = list(map(lambda x: int(x.strftime('%w')), chkin_date))
    df['ci_month'] = list(map(lambda x: int(x.strftime('%m')), chkin_date))
    df['ci_monthday'] = list(map(lambda x: int(x.strftime('%d')), chkin_date))
    del df['srch_ci']
    chkout_date = list(df['srch_co'])
    chkout_date = list(map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'), chkout_date))
    df['co_weekday'] = list(map(lambda x: int(x.strftime('%w')), chkout_date))
    df['co_month'] = list(map(lambda x: int(x.strftime('%m')), chkout_date))
    df['co_monthday'] = list(map(lambda x: int(x.strftime('%d')), chkout_date))
    del df['srch_co']
    return df

"""
Variables
"""
# Sampling rate of the data
samp = 100
# Number of rows read
n_rows = 1e8
# Whether to merge the data
merge = False
# sample_train filename, None if not required
train_file = 'input/train_samp_%d_merged.csv' % samp
classifier = RandomForestClassifier(n_estimators=100, max_depth=30, max_features=0.3)

"""
Read data
"""
# Read destinations
destinations = pd.DataFrame.from_csv('input/destinations.csv')
# print(destinations)

# Read and sample train
with open('input/train.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    columns = spamreader.__next__()
    print('Read the train table columns')
    train_samp = []
    for i, row in enumerate(spamreader):
        if n_rows < i:
            break
        if i % samp == 0:
            train_samp.append(row)
        if i % 1e6 == 0:
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

# Read test
test = pd.read_csv('input/test.csv', index_col=0)

"""
Feature engineering
"""
del train_samp['user_id']
del test['user_id']

# Change NaN dates to '1970-01-01'
train_samp[['date_time', 'srch_ci', 'srch_co']] = train_samp[['date_time',
                                                              'srch_ci', 'srch_co']].replace('', '1970-01-01')
test[['date_time', 'srch_ci', 'srch_co']] = test[['date_time', 'srch_ci', 'srch_co']].replace('', '1970-01-01')

# Remove NaNs
train_samp = train_samp.replace('', '9999')
test = test.replace('', '9999')

# Parse date
train_samp = parse_dates(train_samp)
# test = parse_dates(test)

# Debug printing
print(train_samp.columns.values)

"""
MLing
"""
X_train, X_test, y_train, y_test = train_test_split(train_samp.values, target.values, test_size=0.33, random_state=42)
classifier.fit(X_train, y_train)
train_predict_prob = classifier.predict_proba(X_test)
print(np.sum(y_test == classifier.predict(X_test)) / y_test.shape[0])
