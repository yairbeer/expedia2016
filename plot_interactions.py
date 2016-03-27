from functions import *

target_col = 'target'

""" Load data and change into used format"""
print('Load data')
train_raw = pd.read_csv("numerai_training_data.csv")
target = train_raw[target_col]
print(target.value_counts(normalize=True))
train_raw = np.array(train_raw.drop(target_col, axis=1))

train0 = train_raw[target.values == 0, :]
train1 = train_raw[target.values == 1, :]

n_interactions = train_raw.shape[1]
n_rows = int(np.sqrt(n_interactions))
for i in range(train_raw.shape[1]):
    plt.figure(1)
    for j in range(i+1, train_raw.shape[1]):
        plt.subplot(n_interactions / n_rows + 1, n_rows, j)
        np.random.choice(train_raw.shape[0], 50, replace=False)
        plt.plot(train0[:, i], train0[:, j], 'ro',
                 train1[:, i], train1[:, j], 'go')
        plt.xlabel('feature %d' % i)
        plt.ylabel('feature %d' % j)
    plt.show()
