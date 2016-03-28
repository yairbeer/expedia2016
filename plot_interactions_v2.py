from functions import *
from sklearn.ensemble import RandomForestClassifier


def calc_density(arr_i, arr_j, res):
    density = np.zeros((res, res))
    step = 1.0 / res
    for step_i in range(res):
        for step_j in range(res):
            cond_i = np.logical_and(arr_i >= (step_i * step), arr_i < ((step_i + 1) * step))
            cond_j = np.logical_and(arr_j >= (step_j * step),
                                    arr_j < ((step_j + 1) * step))
            if (step_i + 1) * step == 1:
                cond_i = np.logical_or(cond_i, arr_i == 1)
            if (step_j + 1) * step == 1:
                cond_j = np.logical_or(cond_j, arr_j == 1)
            density[step_i, step_j] = np.sum(np.logical_and(cond_i, cond_j))
    density /= np.sum(density)
    return density

train_data = pd.read_csv('numerai_training_data.csv')
train_target = train_data.target.values
del train_data['target']

calc_interactions(train_data, train_target, 'int_data', RandomForestClassifier(n_estimators=10, max_depth=6))

train_data = np.array(train_data)

# Split into 0, 1 trainsets
train0 = train_data[train_target == 0, :]
train1 = train_data[train_target == 1, :]

# For each feature
n_rows = int(np.sqrt(train_data.shape[1]))
# plt.figure(1)
# for i in range(train_data.shape[1]):
#     print('feature %d' % i)
#     print(pd.Series(train0[:, i]).value_counts(normalize=True).head())
#     print(pd.Series(train1[:, i]).value_counts(normalize=True).head())
#     plt.subplot(train_data.shape[1] / n_rows, n_rows, i)
#     # Create density map
#     plt.hist([train0[:, i], train1[:, i]], normed=True)
#     plt.title('%d' % i)
# plt.show()

# *, +, - interactions
for i in range(train_data.shape[1]):
    # plt.figure(1)
    # for j in range(i + 1, train_data.shape[1]):
    #     plt.subplot(train_data.shape[1] / n_rows, n_rows, j)
    #     plt.hist([train0[:, i] + train0[:, j], train1[:, i] + train1[:, j]], normed=True)
    #     plt.title('%d + %d' % (i, j))
    # plt.show()
    # plt.figure(1)
    # for j in range(i + 1, train_data.shape[1]):
    #     plt.subplot(train_data.shape[1] / n_rows, n_rows, j)
    #     plt.hist([train0[:, i] - train0[:, j], train1[:, i] - train1[:, j]], normed=True)
    #     plt.title('%d - %d' % (i, j))
    # plt.show()
    plt.figure(1)
    for j in range(i + 1, train_data.shape[1]):
        plt.subplot(train_data.shape[1] / n_rows, n_rows, j)
        plt.hist([train0[:, i] * train0[:, j], train1[:, i] * train1[:, j]], normed=True)
        plt.title('%d * %d' % (i, j))
    plt.show()

# 2D plot for each interaction
resolution = 30
for i in range(train_data.shape[1]):
    plt.figure(1)
    for j in range(i + 1, train_data.shape[1]):
        # plt.subplot(211)
        # plt.hist2d(train0[:, i], train0[:, j], bins=40)
        # plt.subplot(212)
        # plt.hist2d(train1[:, i], train1[:, j], bins=40)
        # Create density map
        train0_int = np.histogram2d(train0[:, i], train0[:, j], bins=40, normed=True)[0]
        train1_int = np.histogram2d(train1[:, i], train1[:, j], bins=40, normed=True)[0]
        plt.subplot(train_data.shape[1] / n_rows, n_rows, j)
        plt.imshow((train1_int - train0_int), origin='lower')
        plt.title('%d, %d' % (i, j))
    plt.show()
    for j in range(i + 1, train_data.shape[1]):
        # plt.subplot(211)
        # plt.hist2d(train0[:, i], train0[:, j], bins=40)
        # plt.subplot(212)
        # plt.hist2d(train1[:, i], train1[:, j], bins=40)
        # Create density map
        train0_int = np.histogram2d(train0[:, i], train0[:, i] + train0[:, j], bins=40, normed=True)[0]
        train1_int = np.histogram2d(train1[:, i], train1[:, i] + train1[:, j], bins=40, normed=True)[0]
        plt.subplot(train_data.shape[1] / n_rows, n_rows, j)
        plt.imshow((train1_int - train0_int), origin='lower')
        plt.title('%d + %d' % (i, j))
    plt.show()
    for j in range(i + 1, train_data.shape[1]):
        # plt.subplot(211)
        # plt.hist2d(train0[:, i], train0[:, j], bins=40)
        # plt.subplot(212)
        # plt.hist2d(train1[:, i], train1[:, j], bins=40)
        # Create density map
        train0_int = np.histogram2d(train0[:, i], train0[:, i] - train0[:, j], bins=40, normed=True)[0]
        train1_int = np.histogram2d(train1[:, i], train1[:, i] - train1[:, j], bins=40, normed=True)[0]
        plt.subplot(train_data.shape[1] / n_rows, n_rows, j)
        plt.imshow((train1_int - train0_int), origin='lower')
        plt.title('%d - %d' % (i, j))
    plt.show()
    for j in range(i + 1, train_data.shape[1]):
        # plt.subplot(211)
        # plt.hist2d(train0[:, i], train0[:, j], bins=40)
        # plt.subplot(212)
        # plt.hist2d(train1[:, i], train1[:, j], bins=40)
        # Create density map
        train0_int = np.histogram2d(train0[:, i], train0[:, i] * train0[:, j], bins=40, normed=True)[0]
        train1_int = np.histogram2d(train1[:, i], train1[:, i] * train1[:, j], bins=40, normed=True)[0]
        plt.subplot(train_data.shape[1] / n_rows, n_rows, j)
        plt.imshow((train1_int - train0_int), origin='lower')
        plt.title('%d * %d' % (i, j))
    plt.show()


