import os

import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import lasagne
# import theano.tensor as T
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

FTRAIN = ['~/Programs/myo/myo-nets/data/emg_data_train_rest.csv',
          '~/Programs/myo/myo-nets/data/emg_data_train_finger1.csv',
          '~/programs/myo/myo-nets/data/emg_data_train_finger2.csv',
          '~/Programs/myo/myo-nets/data/emg_data_train_finger3.csv',
          '~/Programs/myo/myo-nets/data/emg_data_train_finger4.csv',
          '~/Programs/myo/myo-nets/data/emg_data_train_finger5.csv']
FTEST = ['~/Programs/myo/myo-nets/data/emg_data_test_finger3.csv']

ns = 40


def load(fname, isTrain):
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe
    rows_list = []
    for i in xrange(len(df) / ns):
        if isTrain:
            if (df.iloc[i*ns, 8] == 'rest'):
                idx = 0
            elif (df.iloc[i*ns, 8] == 'finger1'):
                idx = 1
            elif (df.iloc[i*ns, 8] == 'finger2'):
                idx = 2
            elif (df.iloc[i*ns, 8] == 'finger3'):
                idx = 3
            elif (df.iloc[i*ns, 8] == 'finger4'):
                idx = 4
            elif (df.iloc[i*ns, 8] == 'finger5'):
                idx = 5
            else:
                idx = -1

            dict1 = {'X': df.iloc[i*ns:i*ns+ns, 0:8].values.flatten(),
                     'Y': np.array([idx])}
        else:
            dict1 = {'X': df.iloc[i*ns:i*ns+ns, 0:8].values.flatten()}

        rows_list.append(dict1)
    df2 = pd.DataFrame(rows_list)

    X = (np.vstack(df2['X'].values) + 128.0) / 256.0  # scale pixel to [0, 1]
    X = X.astype(np.float32)

    y = None
    if isTrain:  # only FTRAIN has any target columns
        y = np.vstack(df2['Y'].values)
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

y = y.T[0, :]
y = y.astype(np.int32)

# l_in = lasagne.layers.InputLayer(X.shape)
# l_hid1 = lasagne.layers.DenseLayer(l_in, num_units=100,
#                                    nonlinearity=lasagne.nonlinearities.rectify,
#                                    W=lasagne.init.GlorotUniform())
# l_out = lasagne.layers.DenseLayer(l_hid1, num_units=2,
#                                   nonlinearity=T.nnet.softmax)

net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 320),
    hidden_num_units=200,  # number of units in hidden layer
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=6,

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=False,  # flag to indicate we're dealing with regression
    max_epochs=5000,  # we want to train this many epochs
    verbose=1,
    )

net1.fit(X, y)

Xtest = None
for filename in FTEST:
    Xe, _ = load(filename, False)
    if Xtest is None:
        Xtest = Xe
    else:
        Xtest = np.vstack((Xtest, Xe))

y_pred = net1.predict(Xtest)
count = 0
for pred in y_pred:
    if pred == 3:
        count += 1
print str(count) + '/' + str(len(y_pred))
print((count * 1.0 / len(y_pred)) * 100)
