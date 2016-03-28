import xgboost
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA, KernelPCA
from functions import *
from scipy.stats import ttest_ind
from sklearn.preprocessing import PolynomialFeatures

poly_transform = PolynomialFeatures(interaction_only=True)

pcaing = PCA(n_components=10)

target_col = 'target'

""" Load data and change into used format"""
print('Load data')
train_raw = pd.read_csv("numerai_training_data.csv")
target = train_raw[target_col]
print(target.value_counts(normalize=True))
train_raw = np.array(train_raw.drop(target_col, axis=1))

test_raw = pd.read_csv("numerai_tournament_data.csv")
test_results = test_raw['t_id']
test_results.index = test_raw['t_id']
test_results = pd.DataFrame(test_results)
test_results['probability'] = np.zeros((test_raw.shape[0]))
del test_results['t_id']
del test_raw['t_id']
test_raw = np.array(test_raw)

print(train_raw)
print(test_raw)

# Get polynomial features
train_poly = poly_transform.fit_transform(train_raw)[:, 1:]
test_poly = poly_transform.transform(test_raw)[:, 1:]
print(train_poly.shape, test_poly.shape)

# t-test columns
train0 = train_poly[target.values == 0, :]
train1 = train_poly[target.values == 1, :]

t_test_scores = []
for col_i in range(train_poly.shape[1]):
    t_test_scores.append(ttest_ind(train0[:, col_i], train1[:, col_i]))
t_test_scores = np.array(t_test_scores)

p_vals = np.log(t_test_scores[:, 1])
"""
CV
"""
best_score = 10
best_params = 0
best_train_prediction = 0
best_prediction = 0
meta_solvers_train = []
meta_solvers_test = []
best_train = 0
best_test = 0

param_grid = [
              {'silent': [1],
               'nthread': [2],
               'booster': ['gblinear'],
               'eval_metric': ['logloss'],
               'eta': [0.01],
               'objective': ['binary:logistic'],
               'num_round': [8000],
               'n_monte_carlo': [5],
               'cv_n': [5],
               'test_rounds_fac': [1.2],
               'mc_test': [False],
               'pca_n': [10],
               'p_thresh': [-35, -30, -20, -10, -5]
               }
              ]

print('start CV')
early_stopping = 200
mc_round_list = []
mc_logloss_mean = []
mc_logloss_sd = []
params_list = []
print_results = []
for params in ParameterGrid(param_grid):

    train = train_poly[:, p_vals < params['p_thresh']]
    test = test_poly[:, p_vals < params['p_thresh']]
    print('There are %d columns' % train.shape[1])

    # CV
    mc_auc = []
    mc_round = []
    mc_train_pred = []
    for i_mc in range(params['n_monte_carlo']):
        cv_n = params['cv_n']
        kf = StratifiedKFold(target.values, n_folds=cv_n, shuffle=True, random_state=i_mc ** 3)

        xgboost_rounds = []
        train_predictions = np.zeros((train.shape[0],))

        for cv_train_index, cv_test_index in kf:
            X_train, X_test = train[cv_train_index, :], train[cv_test_index, :]
            y_train, y_test = target.iloc[cv_train_index].values, target.iloc[cv_test_index].values

            # train machine learning
            xg_train = xgboost.DMatrix(X_train, label=y_train)
            xg_test = xgboost.DMatrix(X_test, label=y_test)

            watchlist = [(xg_train, 'train'), (xg_test, 'test')]

            num_round = params['num_round']
            xgclassifier = xgboost.train(params, xg_train, num_round, watchlist, early_stopping_rounds=early_stopping);
            xgboost_rounds.append(xgclassifier.best_iteration)

        num_round = int(np.mean(xgboost_rounds))
        print('The best n_rounds is %d' % num_round)

        for cv_train_index, cv_test_index in kf:
            X_train, X_test = train[cv_train_index, :], train[cv_test_index, :]
            y_train, y_test = target.iloc[cv_train_index].values, target.iloc[cv_test_index].values

            # train machine learning
            xg_train = xgboost.DMatrix(X_train, label=y_train)
            xg_test = xgboost.DMatrix(X_test, label=y_test)

            watchlist = [(xg_train, 'train'), (xg_test, 'test')]

            xgclassifier = xgboost.train(params, xg_train, num_round, watchlist);

            # predict
            predicted_results = xgclassifier.predict(xg_test)
            train_predictions[cv_test_index] = predicted_results

        print('AUC score ', log_loss(target.values, train_predictions))
        mc_auc.append(log_loss(target.values, train_predictions))
        mc_train_pred.append(train_predictions)
        mc_round.append(num_round)

    mc_train_pred = np.mean(np.array(mc_train_pred), axis=0)

    mc_round_list.append(int(np.mean(mc_round)))
    mc_logloss_mean.append(np.mean(mc_auc))
    mc_logloss_sd.append(np.std(mc_auc))
    print('The AUC range is: %.5f to %.5f and best n_round: %d' %
          (mc_logloss_mean[-1] - mc_logloss_sd[-1], mc_logloss_mean[-1] + mc_logloss_sd[-1], mc_round_list[-1]))
    print_results.append('The AUC range is: %.5f to %.5f and best n_round: %d' %
                         (mc_logloss_mean[-1] - mc_logloss_sd[-1], mc_logloss_mean[-1] + mc_logloss_sd[-1], mc_round_list[-1]))
    print('For ', mc_auc)
    print('The AUC of the average prediction is: %.5f' % log_loss(target.values, mc_train_pred))
    meta_solvers_train.append(mc_train_pred)

    # train machine learning
    xg_train = xgboost.DMatrix(train, label=target.values)
    xg_test = xgboost.DMatrix(test)

    if params['mc_test']:
        watchlist = [(xg_train, 'train')]

        num_round = int(mc_round_list[-1] * params['test_rounds_fac'])
        mc_pred = []
        for i_mc in range(params['n_monte_carlo']):
            params['seed'] = i_mc
            xg_train = xgboost.DMatrix(train, label=target.values)
            xg_test = xgboost.DMatrix(test)

            watchlist = [(xg_train, 'train')]

            xgclassifier = xgboost.train(params, xg_train, num_round, watchlist);
            mc_pred.append(xgclassifier.predict(xg_test))

        meta_solvers_test.append(np.mean(np.array(mc_pred), axis=0))

        """ Write opt solution """
        print('writing to file')
        pd.DataFrame(mc_train_pred).to_csv('train_xgboost_lin_int_%d.csv' % params['p_thresh'])
        test_results['probability'] = meta_solvers_test[-1]
        test_results.to_csv("test_xgboost_lin_fac12_int_%d.csv" % params['p_thresh'])

    if mc_logloss_mean[-1] < best_score:
        print('new best log loss')
        best_score = mc_logloss_mean[-1]
        best_params = params
        best_train_prediction = mc_train_pred
        if params['mc_test']:
            best_prediction = meta_solvers_test[-1]

print(best_score)
print(best_params)

print(params_list)
print(print_results)
print(mc_logloss_mean)
print(mc_logloss_sd)

""" n_monte_carlo = 5, CV = 5 """
# raw dataset + PCA n=10: 0.69152999790286596 / 0.69014
# raw dataset + PCA n=10 + rbf kernel PCA n=10: nope
# raw dataset + PCA n=10 + m_interactions: nope
