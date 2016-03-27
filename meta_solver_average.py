import pandas as pd
import numpy as np
import glob
from sklearn.metrics import log_loss
from scipy.optimize import minimize


def to_log(arr):
    return np.log(arr)


def to_lin(arr):
    return np.exp(arr)


def opt_weights(weights, *args):
    arr, target = args
    pred = np.average(arr, axis=1, weights=weights)
    return log_loss(target, pred)


def opt_weights_log(weights, *args):
    arr, target = args
    arr = to_log(arr)
    pred = np.average(arr, axis=1, weights=weights)
    pred = to_lin(pred)
    return log_loss(target, pred)

test = pd.read_csv("numerai_tournament_data.csv")
test_results = test['t_id']
test_results.index = test['t_id']
test_results = pd.DataFrame(test_results)
test_results['probability'] = np.zeros((test.shape[0]))
del test_results['t_id']
del test

train = pd.read_csv("numerai_training_data.csv")
print(train.target.value_counts(normalize=True))
basecase = 0.507313
target = train['target'].values

meta_train_solvers = glob.glob('solvers/train*')
meta_train_solvers = sorted(meta_train_solvers)
print(meta_train_solvers)

for i, meta_predict in enumerate(meta_train_solvers):
    meta_train_solvers[i] = pd.DataFrame.from_csv(meta_predict).values
    print(log_loss(target, meta_train_solvers[i]))
meta_train_solvers = np.hstack(tuple(meta_train_solvers))
meta_train_solvers = np.hstack(tuple([meta_train_solvers, np.ones((meta_train_solvers.shape[0], 1)) * basecase]))
meta_train_solvers = meta_train_solvers[:, :-1]
print(meta_train_solvers)

x0 = np.ones((meta_train_solvers.shape[1]))
res = minimize(opt_weights, x0, args=(meta_train_solvers, target), method='Nelder-Mead',
               options={'disp': True})
opt_weights = res.x
train_solver = np.average(meta_train_solvers, axis=1, weights=opt_weights)
print(res.x, log_loss(target, train_solver))

meta_solvers = glob.glob('solvers/test*')
meta_solvers = sorted(meta_solvers)
print(meta_solvers)

for i, meta_predict in enumerate(meta_solvers):
    meta_solvers[i] = pd.DataFrame.from_csv(meta_predict).values
meta_solvers = np.hstack(tuple(meta_solvers))
meta_solvers = np.hstack(tuple([meta_solvers, np.ones((meta_solvers.shape[0], 1)) * basecase]))
meta_solvers = meta_solvers[:, :-1]
# meta_solvers = np.log(meta_solvers)
solver = np.average(meta_solvers, axis=1,
                    weights=opt_weights
                    )
# solver = np.exp(solver)

print('writing to file')
print(solver)
pd.DataFrame(train_solver).to_csv('train_avg_solver.csv')
test_results['probability'] = solver
test_results.to_csv("test_avg_solver.csv")
