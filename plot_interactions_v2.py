import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calc_density(feature_i, feature_j, arr, res):
    density = np.zeros((res, res))
    step = 1.0 / res
    for step_i in range(res):
        for step_j in range(res):
            cond_i = np.logical_and(arr[:, feature_i] >= (step_i * step),
                                    arr[:, feature_i] < ((step_i + 1) * step))
            cond_j = np.logical_and(arr[:, feature_j] >= (step_j * step),
                                    arr[:, feature_j] < ((step_j + 1) * step))
            if (step_i + 1) * step == 1:
                cond_i = np.logical_or(cond_i, arr[:, feature_i] == 1)
            if (step_j + 1) * step == 1:
                cond_j = np.logical_or(cond_j, arr[:, feature_j] == 1)
            density[step_i, step_j] = np.sum(np.logical_and(cond_i, cond_j))
    density /= np.sum(density)
    return density

train_data = pd.read_csv('numerai_training_data.csv')
train_target = train_data.target.values
del train_data['target']
train_data = np.array(train_data)

# Split into 0, 1 trainsets
train0 = train_data[train_target == 0, :]
train1 = train_data[train_target == 1, :]

# For each feature
n_rows = int(np.sqrt(train_data.shape[1]))
plt.figure(1)
for i in range(train_data.shape[1]):
    plt.subplot(train_data.shape[1] / n_rows, n_rows, i)
    # Create density map
    plt.hist([train0[:, i], train1[:, i]], normed=True)
    plt.title('%d' % i)
plt.show()

# For each interaction
resolution = 30
n_rows = int(np.sqrt(train_data.shape[1]))
for i in range(train_data.shape[1]):
    plt.figure(1)
    for j in range(i + 1, train_data.shape[1]):
        plt.subplot(train_data.shape[1] / n_rows, n_rows, j)
        # Create density map
        density_map0 = calc_density(i, j, train0, resolution)
        density_map1 = calc_density(i, j, train1, resolution)
        plt.imshow((density_map1 - density_map0), origin='lower')
        plt.title('%d, %d' % (i, j))
        plt.colorbar(ticks=[-0.001, 0, 0.001])
    plt.show()
