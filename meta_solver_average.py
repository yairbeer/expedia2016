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

meta_train_solvers = glob.glob('train_results/*')
meta_train_solvers = sorted(meta_train_solvers)
# meta_train_solvers = meta_train_solvers[1:]
print(meta_train_solvers)

train = pd.read_csv("input/train.csv")
print(train.target.value_counts())
basecase = 87021.0 / (87021 + 27300)
target = train['target'].values

for i, meta_predict in enumerate(meta_train_solvers):
    meta_train_solvers[i] = pd.DataFrame.from_csv(meta_predict).values
    print(log_loss(target, meta_train_solvers[i]))
meta_train_solvers = np.hstack(tuple(meta_train_solvers))
meta_train_solvers = np.hstack(tuple([meta_train_solvers, np.ones((meta_train_solvers.shape[0], 1)) * basecase]))

res = minimize(opt_weights_log, [1, 1, 1, 1, 1, 1, 1], args=(meta_train_solvers, target), method='Nelder-Mead',
               options={'disp': True})
opt_weights = res.x
solver = np.average(meta_train_solvers, axis=1, weights=opt_weights)
print(res.x, log_loss(target, solver))

meta_solvers = glob.glob('results/*')
meta_solvers = sorted(meta_solvers)
# meta_solvers = meta_solvers[1:]
print(meta_solvers)

for i, meta_predict in enumerate(meta_solvers):
    meta_solvers[i] = pd.DataFrame.from_csv(meta_predict).values
meta_solvers = np.hstack(tuple(meta_solvers))
meta_solvers = np.hstack(tuple([meta_solvers, np.ones((meta_solvers.shape[0], 1)) * basecase]))
meta_solvers = np.log(meta_solvers)
solver = np.average(meta_solvers, axis=1,
                    weights=opt_weights
                    )
solver = np.exp(solver)

print('writing to file')
submission_file = pd.DataFrame.from_csv("sample_submission.csv")
submission_file['PredictedProb'] = solver

submission_file.to_csv("w_average.csv")
