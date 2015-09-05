import os

import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

FTRAIN = ['~/Programs/myo/myo-nets/data/emg_data_rest.csv',
          '~/Programs/myo/myo-nets/data/emg_data_finger3.csv']

ns = 40

def load(fname, isTrain):
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe
    rows_list = []
    for i in xrange(len(df) / ns):
        dict1 = {'X': df.iloc[i*ns:i*ns+ns, 0:8].values.flatten(),
                 'Y': df.iloc[i*ns, 8]}

        rows_list.append(dict1)
    df2 = pd.DataFrame(rows_list)

    X = (np.vstack(df2['X'].values) + 128.0) / 256.0  # scale pixel to [0, 1]
    X = X.astype(np.float32)

    y = None
    if isTrain:  # only FTRAIN has any target columns
        y = df2[df2.columns[-1:]].values
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
    return X, y

X = None
y = None
for filename in FTRAIN:
    Xe, ye = load(filename, True)
    if X is None:
        X = Xe
        y = ye
    else:
        X = np.concatenate((X, Xe))
        y = np.concatenate((y, ye))
