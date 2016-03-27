import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def calc_interactions(df, target, train_loc, destination, classifier):
    """ Calculate *, +, - interaction matrixes
    :param df: dataframe
    :param target: target column
    :param train_loc: index location of indexes
    :param destination: output file destination
    """
    target = np.array(target)
    cols = df.columns.values
    print(cols)

    df_interactions_0 = np.zeros((df.shape[1], df.shape[1]))
    df_interactions_mult = np.zeros((df.shape[1], df.shape[1]))
    df_interactions_add = np.zeros((df.shape[1], df.shape[1]))
    df_interactions_sub = np.zeros((df.shape[1], df.shape[1]))
    df_train = df.loc[train_loc]
    for col_i in np.arange(len(cols)):
        for col_j in np.arange(col_i, len(cols)):
            print('The columns %s and %s' % (cols[col_i], cols[col_j]))
            metric_0 = []
            metric_int_mult = []
            metric_int_add = []
            metric_int_minus = []
            for i_mc in range(5):
                cv_n = 2
                kf = StratifiedKFold(target, n_folds=cv_n, shuffle=True, random_state=i_mc ** 3)
                train_i = np.zeros((df_train.shape[0], 3))
                train_i[:, 0] = df_train[cols[col_i]].values
                train_i[:, 1] = df_train[cols[col_j]].values
                class_pred = np.zeros(target.shape)
                for train_index, test_index in kf:
                    X_train, X_test = train_i[train_index, :], train_i[test_index, :]
                    y_train, y_test = target[train_index], target[test_index]

                    classifier.fit(X_train, y_train)

                    # predict
                    class_pred[test_index] = classifier.predict_proba(X_test)[:, 1]

                # evaluate
                metric_0.append(roc_auc_score(target, class_pred))

                train_i[:, 2] = df_train[cols[col_i]].values * df_train[cols[col_j]].values
                if np.max(df_train[cols[col_i]].values * df_train[cols[col_j]].values) > 1e10:
                    train_i = np.clip(train_i, -9999999, 1e10)
                for train_index, test_index in kf:
                    X_train, X_test = train_i[train_index, :], train_i[test_index, :]
                    y_train, y_test = target[train_index], target[test_index]

                    classifier.fit(X_train, y_train)

                    # predict
                    # if cols[col_i] == 'imp_op_var39_comer_ult1' and cols[col_j] == 'var39_prod':
                    #     print(np.max(train_i), np.min(train_i))
                    class_pred[test_index] = classifier.predict_proba(X_test)[:, 1]

                # evaluate
                metric_int_mult.append(roc_auc_score(target, class_pred))

                train_i[:, 2] = df_train[cols[col_i]].values + df_train[cols[col_j]].values
                for train_index, test_index in kf:
                    X_train, X_test = train_i[train_index, :], train_i[test_index, :]
                    y_train, y_test = target[train_index], target[test_index]

                    classifier.fit(X_train, y_train)

                    # predict
                    class_pred[test_index] = classifier.predict_proba(X_test)[:, 1]

                # evaluate
                metric_int_add.append(roc_auc_score(target, class_pred))

                train_i[:, 2] = df_train[cols[col_i]].values - df_train[cols[col_j]].values
                for train_index, test_index in kf:
                    X_train, X_test = train_i[train_index, :], train_i[test_index, :]
                    y_train, y_test = target[train_index], target[test_index]

                    classifier.fit(X_train, y_train)

                    # predict
                    class_pred[test_index] = classifier.predict_proba(X_test)[:, 1]

                # evaluate
                metric_int_minus.append(roc_auc_score(target, class_pred))

            df_interactions_0[col_i, col_j] = np.mean(metric_0)
            df_interactions_mult[col_i, col_j] = np.mean(metric_int_mult)
            df_interactions_add[col_i, col_j] = np.mean(metric_int_add)
            df_interactions_sub[col_i, col_j] = np.mean(metric_int_minus)

            print('AUC score for columns %s and %s is: ref %.4f, prod %.4f, plus %.4f, minus %.4f' %
                  (cols[col_i], cols[col_j], np.mean(metric_0), np.mean(metric_int_mult) - np.mean(metric_0),
                   np.mean(metric_int_add) - np.mean(metric_0), np.mean(metric_int_minus) - np.mean(metric_0)))

    df_interactions_mult = pd.DataFrame(df_interactions_mult, index=cols, columns=cols)
    df_interactions_mult.to_csv(destination + 'abs_mult.csv')
    df_interactions_add = pd.DataFrame(df_interactions_add, index=cols, columns=cols)
    df_interactions_add.to_csv(destination + 'abs_add.csv')
    df_interactions_sub = pd.DataFrame(df_interactions_sub, index=cols, columns=cols)
    df_interactions_sub.to_csv(destination + 'abs_sub.csv')

    df_interactions_mult = pd.DataFrame(df_interactions_mult - df_interactions_0, index=cols, columns=cols)
    df_interactions_mult.to_csv(destination + 'rel_mult.csv')
    df_interactions_add = pd.DataFrame(df_interactions_add - df_interactions_0, index=cols, columns=cols)
    df_interactions_add.to_csv(destination + 'rel_add.csv')
    df_interactions_sub = pd.DataFrame(df_interactions_sub - df_interactions_0, index=cols, columns=cols)
    df_interactions_sub.to_csv(destination + 'rel_sub.csv')


def add_interactions(dataset, prefix, add_threshhold):
    """
    :param dataset: dataframe
    :param prefix: filename prefix
    :param add_threshhold: The threshhold for adding the interaction for the calculation
    :return: train and test dataframes with interactions
    """
    dataset_columns = list(dataset.columns.values)

    df_interactions_mult = pd.DataFrame.from_csv(prefix + '_mult.csv')
    df_interactions_add = pd.DataFrame.from_csv(prefix + '_add.csv')
    df_interactions_sub = pd.DataFrame.from_csv(prefix + '_sub.csv')

    columns = df_interactions_mult.columns.values

    df_interactions_mult = np.array(df_interactions_mult)
    df_interactions_add = np.array(df_interactions_add)
    df_interactions_sub = np.array(df_interactions_sub)

    count = 0
    for i in np.arange(columns.shape[0]):
        for j in np.arange(i, columns.shape[0]):
            if columns[i] in dataset_columns and columns[j] in dataset_columns:
                if df_interactions_mult[i, j] > add_threshhold:
                    new_col = columns[i] + '_' + columns[j] + '_mult'
                    count += 1
                    dataset[new_col] = dataset[columns[i]].values * dataset[columns[j]].values
                if df_interactions_add[i, j] > add_threshhold:
                    new_col = columns[i] + '_' + columns[j] + '_add'
                    count += 1
                    dataset[new_col] = dataset[columns[i]].values + dataset[columns[j]].values
                if df_interactions_sub[i, j] > add_threshhold:
                    new_col = columns[i] + '_' + columns[j] + '_sub'
                    count += 1
                    dataset[new_col] = dataset[columns[i]].values - dataset[columns[j]].values
    print('Added %d columns' % count)
    return dataset


def add_special_int(dataset, interactions_crit):
    """
    add interactions that were found useful
    :param dataset: dataset
    :param interactions_crit: which critical level of interactions to add
    :return: dataset with special interactions
    """
    if interactions_crit > 0:
        tested_cols = []
        for col_name in dataset.columns.values:
            if '13y' in col_name.split('_'):
                tested_cols.append(col_name)
        tested_cols = sorted(tested_cols)
        col_name = '%s_%d_count' % ('13y', 0)
        dataset[col_name] = np.sum(dataset[tested_cols].values == 0, axis=1)

        tested_cols = []
        for col_name in dataset.columns.values:
            if 'num' in col_name.split('_') and len(col_name.split('_')) == 2:
                tested_cols.append(col_name)
        tested_cols = sorted(tested_cols)
        col_name = '%s_sum' % 'num'
        dataset[col_name] = np.sum(dataset[tested_cols].values, axis=1)

        tested_cols = []
        for col_name in dataset.columns.values:
            if 'delta' in col_name.split('_'):
                tested_cols.append(col_name)
        tested_cols = sorted(tested_cols)
        col_name = '%s_%d_count' % ('delta', 0)
        dataset[col_name] = np.sum(dataset[tested_cols].values == 0, axis=1)

        tested_cols = []
        for col_name in dataset.columns.values:
            if 'meses' in col_name.split('_'):
                tested_cols.append(col_name)
        tested_cols = sorted(tested_cols)
        col_name = '%s_%d_count' % ('meses', 0)
        dataset[col_name] = np.sum(dataset[tested_cols].values == 0, axis=1)

    if interactions_crit > 1:
        tested_cols = []
        for col_name in dataset.columns.values:
            if 'num' in col_name.split('_') and len(col_name.split('_')) == 2:
                tested_cols.append(col_name)
        tested_cols = sorted(tested_cols)
        col_name = '%s_%d_count' % ('num', 0)
        dataset[col_name] = np.sum(dataset[tested_cols].values == 0, axis=1)

        tested_cols = []
        for col_name in dataset.columns.values:
            if 'num' in col_name.split('_') and '0' in col_name.split('_'):
                tested_cols.append(col_name)
        tested_cols = sorted(tested_cols)
        col_name = '%s_sum' % 'num_0'
        dataset[col_name] = np.sum(dataset[tested_cols].values, axis=1)

        tested_cols = []
        for col_name in dataset.columns.values:
            if 'num' in col_name.split('_') and 'efect' in col_name.split('_'):
                tested_cols.append(col_name)
        tested_cols = sorted(tested_cols)
        col_name = '%s_%d_count' % ('num_efect', 0)
        dataset[col_name] = np.sum(dataset[tested_cols].values == 0, axis=1)

    return dataset


def calc_feature(df, target, train_loc, destination, classifier):
    """ Calculate *, +, - interaction matrixes
    :param df: dataframe
    :param target: target column
    :param train_loc: index location of indexes
    :param destination: output file destination
    """
    target = np.array(target)
    cols = df.columns.values
    print(cols)

    df_features = np.zeros((df.shape[1], 1))
    df_train = df.loc[train_loc]
    for col_i in np.arange(len(cols)):
        metric_0 = []
        for i_mc in range(10):
            cv_n = 2
            kf = StratifiedKFold(target, n_folds=cv_n, shuffle=True, random_state=i_mc ** 3)
            train_i = np.zeros((df_train.shape[0], 1))
            train_i[:, 0] = df_train[cols[col_i]].values
            class_pred = np.zeros(target.shape)
            for train_index, test_index in kf:
                X_train, X_test = train_i[train_index, 0], train_i[test_index, 0]
                X_train = X_train.reshape((X_train.shape[0], 1))
                X_test = X_test.reshape((X_test.shape[0], 1))
                y_train, y_test = target[train_index], target[test_index]

                classifier.fit(X_train, y_train)

                # predict
                class_pred[test_index] = classifier.predict_proba(X_test)[:, 1]

            # evaluate
            metric_0.append(roc_auc_score(target, class_pred))
        print('The column is %s, the reference roc_auc_score is: %f, with the SD: %f' %
              (cols[col_i], np.mean(metric_0), np.std(metric_0)))
        df_features[col_i, 0] = np.mean(metric_0)

    df_features = pd.DataFrame(df_features, index=cols)
    df_features.to_csv(destination + '.csv')


def remove_features(dataset, prefix, del_threshhold):
    """
    :param dataset: dataframe
    :param prefix: filename prefix
    :param add_threshhold: The threshhold for adding the interaction for the calculation
    :return: train and test dataframes with interactions
    """
    df_interactions_mult = pd.DataFrame.from_csv(prefix + '_mult.csv')
    df_interactions_add = pd.DataFrame.from_csv(prefix + '_add.csv')
    df_interactions_sub = pd.DataFrame.from_csv(prefix + '_sub.csv')

    columns = df_interactions_mult.columns.values

    df_interactions_mult = np.array(df_interactions_mult)
    df_interactions_add = np.array(df_interactions_add)
    df_interactions_sub = np.array(df_interactions_sub)

    count = 0
    for i in np.arange(columns.shape[0]):
        for j in np.arange(i, columns.shape[0]):
            if df_interactions_mult[i, j] > del_threshhold:
                new_col = columns[i] + '_' + columns[j] + '_mult'
                count += 1
                dataset[new_col] = dataset[columns[i]].values * dataset[columns[j]].values
            if df_interactions_add[i, j] > del_threshhold:
                new_col = columns[i] + '_' + columns[j] + '_add'
                count += 1
                dataset[new_col] = dataset[columns[i]].values + dataset[columns[j]].values
            if df_interactions_sub[i, j] > del_threshhold:
                new_col = columns[i] + '_' + columns[j] + '_sub'
                count += 1
                dataset[new_col] = dataset[columns[i]].values - dataset[columns[j]].values
    print('Added %d columns' % count)
    return dataset


def count_nans(df, nan_value, numerical_cols, categorical_cols):
    df_num_nans = df[numerical_cols].values == nan_value
    df_cat_nans = df[categorical_cols].values == nan_value

    df['nan_cat'] = np.sum(df_num_nans, axis=1)
    df['nan_num'] = np.sum(df_cat_nans, axis=1)
    return df


def count_pos_neg(df, numerical_cols):
    df_num_pos = df[numerical_cols].values > 0
    df_num_neg = df[numerical_cols].values < 0

    df['pos_count'] = np.sum(df_num_pos, axis=1)
    df['neg_count'] = np.sum(df_num_neg, axis=1)
    return df


def count_nums(df, numerical_cols, num_list):
    for num in num_list:
        df_num = df[numerical_cols].values == num
        col_name = '%d_count' % num
        df[col_name] = np.sum(df_num, axis=1)
    return df


def interactions_by_colname(df):
    columns = df.columns.values

    columns_splited = []
    for col in columns:
        columns_splited.append(col.split('_'))

    corpus = list(columns_splited)
    corpus = [item for sublist in corpus for item in sublist]
    corpus = np.unique(corpus)

    for word in corpus:
        word_cols = []
        for col_name in columns:
            if word in col_name.split('_'):
                word_cols.append(col_name)
        if len(word_cols) > 1:
            word_df = df[word_cols]
            df[word + '_sum'] = np.sum(word_df.values, axis=1)
            df[word + '_prod'] = np.prod(word_df.values, axis=1)

            df_num_pos = word_df.values > 0
            df_num_neg = word_df.values < 0
            df_num_zero = word_df.values == 0

            df[word + '_pos_count'] = np.sum(df_num_pos, axis=1)
            df[word + '_neg_count'] = np.sum(df_num_neg, axis=1)
            df[word + '_zero_count'] = np.sum(df_num_zero, axis=1)
    return df


def expand_primary(string):
    if string == '-1':
        return string
    else:
        return string[0]


def expand_secondary(string):
    if string == '-1':
        return -1
    else:
        if len(string) > 1:
            return string[1]
        else:
            return -2


def expand_cat(df, cat_col):
    name_primary = cat_col + '_A'
    name_secondary = cat_col + '_B'
    # print(df[cat_col])
    df[name_primary] = df[cat_col].map(lambda x: expand_primary(str(x)))
    df[name_secondary] = df[cat_col].map(lambda x: expand_secondary(str(x)))
    # print(df[name_primary])
    # print(df[name_secondary])
    df.drop(cat_col, axis=1)
    return df


def chi_columns(df_train, chi_val, chi_vec, freedom_thresh):
    true_columns = chi_vec > chi_val
    for i, feature in enumerate(df_train.columns.values):
        n_degrees_of_freedom = df_train[feature].value_counts().shape[0]
        if n_degrees_of_freedom > freedom_thresh:
            true_columns[i] = True
    return true_columns


def pval_columns(df_train, pval_thresh, pval_vec, freedom_thresh, add_nan):
    true_columns = pval_vec < pval_thresh
    # print(true_columns)
    for i, feature in enumerate(df_train.columns.values):
        # n_degrees_of_freedom = df_train[feature].value_counts().shape[0]
        # if n_degrees_of_freedom <= freedom_thresh:
        #     true_columns[i] = True
        if add_nan and np.isnan(pval_vec[i]):
            true_columns[i] = True
    return true_columns


def split_pos_neg(df):
    for col_name in df.columns.values:
        cur_col = df[col_name]
        if np.sum(cur_col.values < 0):
            if np.sum(cur_col.values > 0):
                print('%s splitted' % col_name)
                name_pos = col_name + 'pos'
                df[name_pos] = np.zeros(df[col_name].shape)
                df[name_pos].iloc[cur_col.values > 0] = df[col_name].iloc[cur_col.values > 0]

                name_neg = col_name + 'neg'
                df[name_neg] = np.zeros(df[col_name].shape)
                df[name_neg].iloc[cur_col.values < 0] = df[col_name].iloc[cur_col.values < 0]
                df[name_neg].iloc[:] = np.abs(df[name_neg].values)
                # print(df[name_pos].value_counts())
                # print(df[name_neg].value_counts())
            else:
                print('%s got absed' % col_name)
                df[col_name].iloc[:] = np.abs(df[col_name].values)
    return df


def log_col_chk(df, train_loc, target_col, classifier):
    # Only do it for positive numbers
    cv_n = 2
    df_train = df.loc[train_loc]
    for col in df.columns.values:
        if not np.sum(df[col].values < 0):
            # print(col)
            kf = StratifiedKFold(target_col, n_folds=cv_n, shuffle=True, random_state=42)
            class_pred = np.zeros(target_col.shape)
            for train_index, test_index in kf:
                X_train, X_test = df_train[col].values[train_index], df_train[col].values[test_index]
                X_train = X_train.reshape((X_train.shape[0], 1))
                X_test = X_test.reshape((X_train.shape[0], 1))
                y_train, y_test = target_col.iloc[train_index].values, target_col.iloc[test_index].values

                classifier.fit(X_train, y_train)

                # predict
                class_pred[test_index] = classifier.predict_proba(X_test)[:, 1]

            # evaluate
            metric_lin = roc_auc_score(target_col, class_pred)
            # print('The reference roc_auc_score is:', metric_lin)

            kf = StratifiedKFold(target_col, n_folds=cv_n, shuffle=True, random_state=42)
            class_pred = np.zeros(target_col.shape)
            for train_index, test_index in kf:
                X_train, X_test = df_train[col].values[train_index], df_train[col].values[test_index]
                X_train = X_train.reshape((X_train.shape[0], 1))
                X_test = X_test.reshape((X_train.shape[0], 1))
                X_train, X_test = np.log(X_train + 1), np.log(X_test + 1)
                y_train, y_test = target_col.iloc[train_index].values, target_col.iloc[test_index].values

                classifier.fit(X_train, y_train)

                # predict
                class_pred[test_index] = classifier.predict_proba(X_test)[:, 1]

            # evaluate
            metric_log = roc_auc_score(target_col, class_pred)
            # print('The log roc_auc_score is:', metric_log)
            if metric_log > metric_lin:
                # print('The lin AUC was %f, the log AUC was %f, switching to log, the delta is %f'
                #       % (metric_lin, metric_log, metric_log - metric_lin))
                df[col].iloc[:] = np.log(df[col].values + 1)
    return df


def weighted_mean_result(meta_train, meta_test, target):
    def norm_percent(pred_percent):
        # pred_percent = np.clip(pred_percent, 0, 1)
        return pred_percent

    def opt_weights(weights, *args):
        arr, target = args
        pred = np.average(arr, axis=1, weights=weights)
        pred = norm_percent(pred)
        return -1 * roc_auc_score(target, pred)

    x0 = list(np.ones((meta_train.shape[1],)))
    res = minimize(opt_weights, x0, args=(meta_train.copy(), target), method='Nelder-Mead',
                   options={'disp': True})
    opt_weights_list = res.x
    train_solver = np.average(meta_train, axis=1, weights=opt_weights_list)
    print(res.x, roc_auc_score(target, train_solver))

    solver = np.average(meta_test, axis=1,
                        weights=opt_weights_list
                        )
    # solver = norm_percent(solver)
    train_solver = np.average(meta_train, axis=1,
                              weights=opt_weights_list
                              )
    # train_solver = norm_percent(train_solver)
    return train_solver, solver


def remove_zeros(df, train_loc):
    dropping_cols = []
    for col in df.columns.values:
        if not np.sum(df.loc[train_loc][col].values != 0):
            dropping_cols.append(col)
    print('deletes col ', dropping_cols)
    df = df.drop(dropping_cols, axis=1)
    return df


def remove_correlated(df):
    cols = df.columns.values
    dropping_cols = []
    for col_i in np.arange(len(cols)):
        for col_j in np.arange(col_i + 1, len(cols)):
            cor = np.corrcoef(df[cols[col_i]].values, df[cols[col_j]].values)[0, 1]
            if cor == 1 or cor == -1:
                if not (cols[col_j] in dropping_cols):
                    print('Dropped column %s because of correlation to column %s because of correlation %.5f' %
                          (cols[col_j], cols[col_i], cor))
                    dropping_cols.append(cols[col_j])
    print('deletes col ', dropping_cols)
    df = df.drop(dropping_cols, axis=1)
    return df


def remove_sparse(df, train_loc, sparsity):
    sparsity_train = train_loc.shape[0] * sparsity
    dropping_cols = []
    for col in df.columns.values:
        if np.sum(df.loc[train_loc][col].values != 0) < sparsity_train:
            dropping_cols.append(col)
    print('deletes cols: ', dropping_cols)
    df = df.drop(dropping_cols, axis=1)
    return df


def plot_roc(prob, target, name):
    # Compute ROC curve and ROC area for each class
    print(prob, target)
    fpr, tpr, _ = roc_curve(target, prob)
    roc_auc = auc(fpr, tpr)

    # # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    ##############################################################################
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for %s' % name.split('/')[-1])
    plt.legend(loc="lower right")
    plt.show()
