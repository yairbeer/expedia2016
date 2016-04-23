import csv
import pandas as pd
import numpy as np
import datetime
import scipy.stats as stats
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


def percent2mapk(predict_percent, k):
    predict_map = []
    for i_row, pred_row in enumerate(predict_percent):
        predict_map.append([])
        ranked_row = list(stats.rankdata(pred_row, method='ordinal'))
        for op_rank in range(k):
            predict_map[i_row].append(ranked_row.index(n_classes - op_rank - 1))
    return predict_map


def list2str(predict_list, join_by):
    str_list = []
    for predict_result in predict_list:
        predict_result = list(map(lambda x: str(x), predict_result))
        str_list.append(join_by.join(predict_result))
    return str_list


def y2list(y_array):
    y_list = []
    for actual in y_array:
        y_list.append([actual])
    return y_list

"""
Variables
"""
# Number of classes
n_classes = 100
# Sampling rate of the data
samp = 100
# Number of rows for train
n_rows = 1e4
# Whether to merge the data
merge = False
# sample_train filename, None if not required
train_file = None
# RF classifier properties
classifier = RandomForestClassifier(n_estimators=25, max_depth=40, random_state=42)
# Test batch
test_batch = 10000

"""
Read data
"""
# Read destinations
destinations = pd.DataFrame.from_csv('input/destinations.csv')

# Read and sample train
print('Read the train table columns')
train_rows = 0
with open('input/train.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    columns = spamreader.__next__()
    train_samp = []
    for i, row in enumerate(spamreader):
        if i % samp == 0:
            train_samp.append(row)
            train_rows += 1
            if train_rows == n_rows:
                break
        if i % 1e6 == 0:
            print(i)
csvfile.close()

# build pandas Dataframe
train_samp = pd.DataFrame(train_samp, columns=columns)
train_samp.index = train_samp.user_id
train_samp.hotel_cluster = train_samp.hotel_cluster.astype(int)

# Merge if required
if merge:
    print('Merging')
    train_samp = pd.merge(train_samp, destinations, left_on=train_samp.srch_destination_id.values.astype(int),
                          right_on=destinations.index.values, how='left')

# Export train file
if train_file:
    print('Saving to sampled train to file')
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
print('Feature engineering')
del train_samp['user_id']
del test['user_id']

# Change NaN dates to '1970-01-01'
train_samp[['date_time', 'srch_ci', 'srch_co']] = train_samp[['date_time', 'srch_ci', 'srch_co']].astype(str)
test[['date_time', 'srch_ci', 'srch_co']] = test[['date_time', 'srch_ci', 'srch_co']].astype(str)
train_samp[['date_time', 'srch_ci', 'srch_co']] = train_samp[['date_time', 'srch_ci',
                                                              'srch_co']].replace(['', 'nan', '2161-10-00'],
                                                                                  ['1970-01-01', '1970-01-01',
                                                                                   '1970-01-01'])
test[['date_time', 'srch_ci', 'srch_co']] = test[['date_time', 'srch_ci',
                                                  'srch_co']].replace(['', 'nan', '2161-10-00'], ['1970-01-01',
                                                                                                  '1970-01-01',
                                                                                                  '1970-01-01'])
# Remove NaNs
train_samp = train_samp.replace('', '9999')
test = test.replace('', '9999')
train_samp = train_samp.fillna(9999)
test = test.fillna(9999)

# Parse date
train_samp = parse_dates(train_samp)
test = parse_dates(test)

# Debug printing
# print(train_samp.columns.values)

"""
MLing, CV
"""
print('CV')
X_train, X_test, y_train, y_test = train_test_split(train_samp.values, target.values, test_size=0.33, random_state=42)
classifier.fit(X_train, y_train)
train_predict_prob = np.zeros((X_test.shape[0], n_classes))
for batch_i in np.arange(0, X_test.shape[0], test_batch):
    if (batch_i + test_batch) < X_test.shape[0]:
        train_predict_prob[batch_i: batch_i + test_batch, :] = \
            classifier.predict_proba(X_test[batch_i: batch_i + test_batch, :])
    else:
        train_predict_prob[batch_i:, :] = classifier.predict_proba(X_test[batch_i:, :])
train_predict_prob = percent2mapk(train_predict_prob, 5)
train_predict_map = percent2mapk(train_predict_prob, 5)
y_test_list = y2list(y_test)
print('The mean average precision is %.4f' % mapk(y_test_list, train_predict_map, k=5))
train_predict_str = list2str(train_predict_map, ' ')

"""
MLing
"""
print('Batch predicting test')
classifier.fit(train_samp.values, target.values)

# Freeing memory
del train_samp, target, X_train, X_test, y_train, y_test, train_predict_prob, train_predict_map

test_predict_prob = np.zeros((test.shape[0], n_classes))
for batch_i in np.arange(0, test.shape[0], test_batch):
    if (batch_i + test_batch) < test.shape[0]:
        test_predict_prob[batch_i: batch_i + test_batch,
                          :] = classifier.predict_proba(test.values[batch_i: batch_i + test_batch, :])
    else:
        test_predict_prob[batch_i:, :] = classifier.predict_proba(test.values[batch_i:, :])
test_predict_map = percent2mapk(test_predict_prob, 5)
test_predict_str = list2str(test_predict_map, ' ')

"""
Submitting
"""
submission = pd.DataFrame.from_csv('input/sample_submission.csv')
submission['hotel_cluster'] = test_predict_str
submission.to_csv('rf_sub_withdates.csv')
