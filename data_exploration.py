import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

chunksize = 10 ** 6
for chunk in pd.read_csv('input/train.csv', chunksize=chunksize):
    print(process(chunk))

# print(train.head())
# print(list(train.columns.values))

submission = pd.DataFrame.from_csv('sample_submission.csv')
print(submission.head())
