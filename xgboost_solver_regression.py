import xgboost
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import log_loss
from functions import *

target_col = 'target'

""" Load data and change into used format"""
print('Load data')
train = pd.read_csv("numerai_training_data.csv")
target = train[target_col]
print(target.value_counts(normalize=True))
train = np.array(train.drop(target_col, axis=1))
print(train)

test = pd.read_csv("numerai_tournament_data.csv")
test_results = test['t_id']
test_results.index = test['t_id']
test_results = pd.DataFrame(test_results)
test_results['probability'] = np.zeros((test.shape[0]))
del test_results['t_id']
del test['t_id']
test = np.array(test)
print(test_results)
print(test)


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
               'eta': [0.003],
               'objective': ['binary:logistic'],
               'num_round': [8000],
               'subsample': [1],
               'n_monte_carlo': [5],
               'cv_n': [5],
               'test_rounds_fac': [1.2],
               'count_n': [0],
               'mc_test': [True],
               'special_feng': [0]
               }
              ]

print('start CV')
early_stopping = 100
mc_round_list = []
mc_logloss_mean = []
mc_logloss_sd = []
params_list = []
print_results = []
for params in ParameterGrid(param_grid):
    print(params)
    params_list.append(params)

    train_predictions = np.ones((train.shape[0],))

    print('There are %d columns' % train.shape[1])

    # CV
    mc_auc = []
    mc_round = []
    mc_train_pred = []
    for i_mc in range(params['n_monte_carlo']):
        cv_n = params['cv_n']
        kf = StratifiedKFold(target.values, n_folds=cv_n, shuffle=True, random_state=i_mc ** 3)

        xgboost_rounds = []

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
"""
Final Solution
"""
""" Write opt solution """
if best_params['mc_test']:
    print('writing to file')
    print(best_prediction)
    pd.DataFrame(best_train_prediction).to_csv('xgboost_lin_train_opt.csv')
    test_results['probability'] = best_prediction
    test_results.to_csv("xgboost_lin_fac12_opt.csv")

# raw dataset: 0.6911
