
import matplotlib.pyplot as plt
import seaborn as sb
import biosppy as bp

import numpy as np
import pandas as pd
import os
import pickle
from scipy import interpolate
import utils_classification as uc
from scipy import signal
import pickle
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import time
import pylab as pl
from IPython import display
import numpy as np
from importlib import reload  # Python 3.4+ only.
reload(bp)
import os
import string

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
names = ["MLP"]
classifiers = [MLPClassifier(alpha=1, max_iter=1000)]

from sklearn.model_selection import RepeatedKFold

from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def score_classifier_pred(c, n, train, test, y_train, y_test, show=False):
    all_scores = []

    # Train the classifier
    c.fit(train, y_train.ravel())

    # Predict test data
    y_pred = c.predict(test)
    class_names = ['N', 'A']
    y_test = [class_names.index(yt) for yt in y_test]
    y_pred = [class_names.index(yp) for yp in y_pred]

    score = classification_report(y_test, y_pred, output_dict=True)['weighted avg']
    #score = np.round(accuracy_score(y_pred, y_test)*100,2)

    # Get the classification accuracy
    print(str(n) + " --- Accuracy: " + str(score) + '%')
    print('-----------------------------------------')
    class_names = np.array(['Apnea', 'Normal'])


    # return [np.round(score*100,2), np.round(accuracy*100,2)]
    return score, y_pred

def score_classifier_fusion(cl1, n, train, test, y_train, y_test, show=False):
    all_scores = []
    cl2 = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    cl2_name = 'RF'

    # Train the classifier
    cl1.fit(train, y_train.ravel())
    cl2.fit(train, y_train.ravel())

    # Predict test data
    y_pred = cl1.predict_proba(test)
    y_pred2 = cl2.predict_proba(test)

    all_pred = [[y_pred[i][0]+y_pred2[i][0], y_pred[i][1]+y_pred2[i][1]] for i in range(len(test))]

    y_pred = [cl1.classes_[np.argmax(all_pred[i])] for i in range(len(all_pred))]

    # score = f1_score(y_test, y_pred, average='micro')
    score = accuracy_score(y_test, y_pred)

    # Get the classification accuracy
    print(str(n) + " --- Accuracy: " + str(score) + '%')
    print('-----------------------------------------')
    class_names = np.array(['Apnea', 'Normal'])


    # return [np.round(score*100,2), np.round(accuracy*100,2)]
    return np.round(score * 100, 2), y_pred

def separate_classes_3(vecs, labs, classes=['A', 'N']):
    # take the whole set and separate classes.
    new_vec, new_labs = [], []
    for cl in classes:
        for i in range(0,len(labs),15):
            if i == 0:
                cl_index = [np.arange(i+5,i+10)]
            else:
                cl_index += [np.arange(i+5,i+10)]
        cl_index = np.concatenate(cl_index)
        vec_ = np.array(vecs)[cl_index]
        lab_ = np.array(labs)[cl_index]
        new_vec += [vec_]
        new_labs += [lab_]
    return new_vec, new_labs

def separate_classes_ap(vecs, labs, classes=['N']):
    # take the whole set and separate classes.
    new_vec, new_labs = [], []
    for cl in classes:
        for i in range(0,len(labs),15):
            if i == 0:
                cl_index = [np.concatenate([np.arange(i,i+5),np.arange(i+10,i+15)])]
            else:
                cl_index += [np.concatenate([np.arange(i,i+5),np.arange(i+10,i+15)])]
        cl_index = np.concatenate(cl_index)
        vec_ = np.array(vecs)[cl_index]
        lab_ = np.array(labs)[cl_index]
        new_vec += [vec_]
        new_labs += [lab_]
    return new_vec, new_labs

def classification(c, n, corr_size, label, training_set_x, testing_set_x, training_set_y, testing_set_y):

    train, y_train = separate_classes_3(training_set_x, training_set_y)
    #train, y_train = uc.separate_classes(training_set_x, training_set_y)
    #train_ap, y_train_ap = separate_classes_ap(training_set_x, training_set_y)

    n_train, n_test, n_ytrain, n_ytest = train_test_split(train[0], y_train[0], test_size=.4, random_state=42)

    #training_set_x = train_ap[0]
    #training_set_y = y_train_ap[0]
    loss = 'cosine_proximity'
    activ ='tanh'
    opt ='adam'


    nodes = [500 ,250 ,50]

    with suppress_stdout():
        #n_train = np.array([nt + np.random.normal(0,0.2,len(nt)) for nt in n_train])
        uc.create_autoencoder(n_train, n_train, n_test, n_test, label, loss, activ, opt, nodes = nodes, epochs=50)

    encoder = pickle.load(open(label + '_encoder', 'rb'))
    decoder = pickle.load(open(label + '_decoder', 'rb'))

    enc = encoder.predict(testing_set_x)
    dec = decoder.predict(enc)

    enc_train = encoder.predict(training_set_x)
    dec_train = decoder.predict(enc_train)

    train_cl ,test_cl = [] ,[]
    y_train_new = []
    y_test_new = []
    cs = int(1000 /corr_size)
    for d in range(len(dec_train)):
        corr_train = [pearsonr(dec_train[d][i: i +cs] ,training_set_x[d][i: i +cs])[0] for i in range(0 ,len(dec_train[d]) ,cs)]
        if np.isfinite(corr_train).all():
            train_cl += [corr_train]
            y_train_new += [training_set_y[d]]

    for d in range(len(dec)):

        corr = [pearsonr(dec[d][i: i +cs] ,testing_set_x[d][i: i +cs])[0] for i in range(0 ,len(dec[d]) ,cs)]

        if np.isfinite(corr).all():
            test_cl += [corr]
            y_test_new += [testing_set_y[d]]

    _score, pred_label = score_classifier_pred(c, n, train_cl,test_cl, np.array(y_train_new), np.array(y_test_new), show=False)
    return _score, pred_label

def feature_classification(c, n, corr_size, label, training_set_x, testing_set_x, training_set_y, testing_set_y):
    train, y_train = separate_classes_3(training_set_x, training_set_y)
    # train, y_train = uc.separate_classes(training_set_x, training_set_y)
    # train_ap, y_train_ap = separate_classes_ap(training_set_x, training_set_y)

    n_train, n_test, n_ytrain, n_ytest = train_test_split(train[1], y_train[1], test_size=.4, random_state=42)

    # training_set_x = train_ap[0]
    # training_set_y = y_train_ap[0]
    loss = 'cosine_proximity'
    activ = 'tanh'
    opt = 'adam'

    dec = testing_set_x[:]

    dec_train = training_set_x[:]

    train_cl, test_cl = [], []
    y_train_new = []
    y_test_new = []
    cs = int(1000 / corr_size)
    for d in range(len(dec_train)):
        corr_train = [pearsonr(training_set_x[0][i: i + cs], training_set_x[d][i: i + cs])[0] for i in
                      range(0, len(dec_train[d]), cs)]
        if np.isfinite(corr_train).all():
            train_cl += [corr_train]
            y_train_new += [training_set_y[d]]

    for d in range(len(dec)):

        corr = [pearsonr(testing_set_x[0][i: i + cs], testing_set_x[d][i: i + cs])[0] for i in range(0, len(dec[d]), cs)]

        if np.isfinite(corr).all():
            test_cl += [corr]
            y_test_new += [testing_set_y[d]]

    _score, pred_label = score_classifier_pred(c, n, train_cl, test_cl, np.array(y_train_new), np.array(y_test_new),
                                               show=False)
    return _score, pred_label


user_labs = pickle.load(open('user_labs', 'rb'))
user_data = pickle.load(open('user_data', 'rb'))
aaa

def rgbt_eval(user_labs, user_data, list_corr_size, classifiers_name):
    Y = np.array(user_labs)
    FS_X = np.array(user_data)
    FS_X_train = [0] * 27
    FS_X_test = [0] * 27
    y_test = [0] * 27
    y_train = [0] * 27
    fold = 0

    #classifiers_names = ['KNN', 'SVM', 'Decision Tree', 'Random Forest', 'MLP', 'NB']
    #classifiers = [KNeighborsClassifier(), SVC(gamma=2, C=1, probability=True), DecisionTreeClassifier(),
     #              RandomForestClassifier(), MLPClassifier(), GaussianNB()]
    classifiers = [SVC(gamma=2, C=1, probability=True), MLPClassifier(), GaussianNB()]
    #classifiers = [KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]

    kf = RepeatedKFold(n_splits=9, n_repeats=3, random_state=10)

    for train_index, test_index in kf.split(Y):
        #     print(Y.shape)
        #     print(test_index, train_index)
        FS_X_train[fold] = np.vstack(FS_X[train_index])
        y_train[fold] = np.concatenate(Y[train_index], axis=0)
        FS_X_test[fold] = np.vstack(FS_X[test_index])
        y_test[fold] = np.concatenate(Y[test_index], axis=0)



        fold += 1
    print('---- Train Test Split was concluded ----')


    for c in range(len(classifiers_name)):

        print('---------------- Testing classifier ' + classifiers_name[c])

        for corr_size in list_corr_size:
            fo = 0
            name_cl = classifiers_name[c]
            cl = classifiers[c]
            ji = 0
            label = pd.DataFrame(columns = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

            for fsx in range(len(FS_X_train)):
                print('= ' *10 + str(fo) +'= ' *10)

                signal_train = FS_X_train[fsx]
                _ap_train = y_train[fsx]
                signal_test = FS_X_test[fsx]
                _ap_test = y_test[fsx]
                print(signal_train.shape)

                pickle.dump([signal_train,signal_test],open('data_sep','wb'))

                pickle.dump([_ap_train,_ap_test],open('label_sep','wb'))

                _acc, pred_label = classification(cl, name_cl, corr_size, 'RGBT_0', signal_train, signal_test, _ap_train, _ap_test)

                for i in range(0,len(pred_label),15):
                    label.loc[ji] = pred_label[i:i+15]
                    ji+=1
                if fo == 0:
                    score = pd.DataFrame([_acc], columns=_acc.keys())

                score.loc[fsx] = _acc
                fo+=1
            score.to_csv(r'score_f1_Sinus_'+str(corr_size)+'_autocorr_'+name_cl+'_rgbt.csv ' , index = None, header=True)

classifiers_names = ['SVM','MLP','NB']

rgbt_eval(user_labs, user_data, [4,10,20,50], classifiers_names)

labs = ['Free', 'Sinusoidal', 'Apnea']

def normal(sg, minsg, maxsg):
    res = 100 * (sg - minsg) / (maxsg - minsg)
    return res

import seaborn as sn

def plot_rgbt_data():
    norm_free = normal(user_data[12][2], np.min(user_data[12][2]), np.max(user_data[12][2]))
    norm_sinus = normal(user_data[12][6], np.min(user_data[12][6]), np.max(user_data[12][6]))
    norm_apnea = normal(user_data[12][11], np.min(user_data[12][11]), np.max(user_data[12][11]))
    norm_free1 = normal(user_data[15][2], np.min(user_data[15][2]), np.max(user_data[15][2]))
    norm_sinus1 = normal(user_data[15][6], np.min(user_data[15][6]), np.max(user_data[15][6]))
    norm_apnea1 = normal(user_data[15][11], np.min(user_data[15][11]), np.max(user_data[15][11]))


    sn.set(font_scale=3)
    clr = ['#0066cc', '#66ccff','#339933']
    fig = plt.figure(figsize=(30,10))
    fig.tight_layout()
    fig.suptitle('Unscontrained (Free), Regular (Sinusoidal) and Apnea samples from the BA-RGBT database')

    plt.subplot(2,3,1)
    plt.title(labs[0])
    plt.plot(np.arange(0,60,0.06),norm_free, linewidth=8,color=clr[2])
    plt.ylim(-1,102)
    #plt.xlabel('Time (s)')
    plt.subplot(2,3,2)
    plt.title(labs[1])
    plt.plot(np.arange(0,60,0.06),norm_sinus, linewidth=8,color=clr[2])
    plt.ylim(-1,102)
    #plt.xlabel('Time (s)')
    plt.subplot(2,3,3)
    plt.title(labs[2])
    plt.plot(np.arange(0,60,0.06),norm_apnea, linewidth=8,color=clr[0])

    plt.ylim(-1,102)
    #plt.xlabel('Time (s)')
    ax = plt.subplot(2,3,4)
    plt.plot(np.arange(0,60,0.06),norm_free1, linewidth=8,color=clr[2])
    plt.ylim(-1,102)
    ax.yaxis.set_label_coords(-0.25, 1)
    plt.ylabel('Amplitude (%)', va='top')
    plt.xlabel('Time (s)')
    ax = plt.subplot(2,3,5)
    plt.plot(np.arange(0,60,0.06),norm_sinus1, linewidth=8,color=clr[2])
    plt.ylim(-1,102)
    plt.xlabel('Time (s)')

    ax.yaxis.set_label_coords(-0.25, 1)
    plt.ylabel('Amplitude (%)', va='top')
    ax = plt.subplot(2,3,6)
    plt.plot(np.arange(0,60,0.06),norm_apnea1, linewidth=8,color=clr[0])
    plt.ylim(-1,102)
    plt.xlabel('Time (s)')

    ax.yaxis.set_label_coords(-0.25, 1)
    plt.ylabel('Amplitude (%)', va='top')
    fig.subplots_adjust(left=None, bottom=None, right=None, top=0.8, wspace=0.3, hspace=None)

    plt.savefig('signal_apnea.png', bbox_inches='tight', format='png')
    plt.show()

plot_rgbt_data()