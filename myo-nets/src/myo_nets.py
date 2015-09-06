import os

import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import lasagne
import theano.tensor as T
# from lasagne import layers
# from lasagne.updates import nesterov_momentum
# from nolearn.lasagne import NeuralNet

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
        X = np.vstack((X, Xe))
        y = np.vstack((y, ye))

l_in = lasagne.layers.InputLayer(X.shape)
l_hid1 = lasagne.layers.DenseLayer(l_in, num_units=100,
                                   nonlinearity=lasagne.nonlinearities.rectify,
                                   W=lasagne.init.GlorotUniform())
l_out = lasagne.layers.DenseLayer(l_hid1, num_units=2,
                                  nonlinearity=T.nmet.softmax)

# net1 = NeuralNet(
#     layers=[  # three layers: one hidden layer
#         ('input', layers.InputLayer),
#         ('hidden', layers.DenseLayer),
#         ('output', layers.DenseLayer),
#         ],
#     # layer parameters:
#     input_shape=(None, 1, X.shape[0], X.shape[1]),
#     hidden_num_units=100,  # number of units in hidden layer
#     output_nonlinearity=lasagne.nonlinearities.softmax,
#     output_num_units=len(y),
#
#     # optimization method:
#     update=nesterov_momentum,
#     update_learning_rate=0.01,
#     update_momentum=0.9,
#
#     regression=False,  # flag to indicate we're dealing with regression
#     max_epochs=400,  # we want to train this many epochs
#     verbose=1,
#     )
#
# net1.fit(X, y)
