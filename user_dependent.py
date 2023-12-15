
import time
import pylab as pl
from IPython import display
import pickle
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
import numpy as np
import tsfel as tsfel

def user_dep(train_label, test_label, decode = False, features = False):
    X = np.array(pickle.load(open(train_label + '_X', 'rb')))
    Y = np.array(pickle.load(open(train_label + '_Y', 'rb')))
    from sklearn.model_selection import KFold
    if decode:
        encoder_ap = pickle.load(open('apneia_encoder', 'rb'))
        FS_X = np.array([encoder_ap.predict(np.array(X[xi]).reshape(len(X[xi]),1000)) for xi in range(len(X))])
    if features:
        googleSheet_name = "Mariana"
        cfg_file = tsfel.extract_sheet(googleSheet_name)
        FS_X = np.array([np.array(tsfel.extract_features(xi, 'x', cfg_file, fs=100.0)) for xi in X])
    else:
        FS_X = np.array([np.array(X[xi]).reshape(len(X[xi]),1000) for xi in range(len(X))])

    print([FS_X[fs].shape for fs in range(len(FS_X))])
    with open(r"FS_X.pickle", "wb") as output_file:
       pickle.dump(FS_X, output_file)

    FS_X = np.array(pickle.load(open('FS_X.pickle','rb')))
    FS_X_train = np.array(FS_X)
    Y_train = np.array(Y)

    with open(r"FS_X_train.pickle", "wb") as output_file:
        pickle.dump(FS_X_train, output_file)
    with open(r"y_train.pickle", "wb") as output_file:
        pickle.dump(Y_train, output_file)

    X = np.array(pickle.load(open(test_label + '_X', 'rb')))
    Y = np.array(pickle.load(open(test_label + '_Y', 'rb')))
    if decode:
        encoder_ap = pickle.load(open('bre_encoder', 'rb'))
        FS_X = np.array([encoder_ap.predict(np.array(X[xi]).reshape(len(X[xi]),1000)) for xi in range(len(X))])
    if features:
        googleSheet_name = "Mariana"
        cfg_file = tsfel.extract_sheet(googleSheet_name)
        FS_X = np.array([np.array(tsfel.extract_features(xi, 'x', cfg_file, fs=100.0)) for xi in X])

    else:
        FS_X = np.array([np.array(X[xi]).reshape(len(X[xi]),1000) for xi in range(len(X))])
    print([FS_X[fs].shape for fs in range(len(FS_X))])
    FS_X_test = np.array(FS_X)
    Y_test = np.array(Y)



    with open(r"FS_X_test.pickle", "wb") as output_file:
        pickle.dump(FS_X_test, output_file)

    with open(r"y_test.pickle", "wb") as output_file:
        pickle.dump(Y_test, output_file)

user_dep('ol','ol', decode=False, features=False)