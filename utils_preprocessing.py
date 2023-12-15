
import os
import pickle

import biosppy as bp
import numpy as np
import pandas as pd

from scipy import signal


def normal(sg, minsg, maxsg):
    res = 2 * (sg - minsg) / (maxsg - minsg) - 1
    return res


def find_extremes(sig, mode, th):
    indexes, values = bp.tools.find_extrema(sig, mode)
    ind, peaks = [], []
    for pk in range(len(values)):
        if abs(values[pk] - np.mean(sig)) > float(th):
            ind += [indexes[pk]]
            peaks += [values[pk]]
    return ind, peaks


def dist(sig):
    """ Calculates the total distance traveled by the signal,
    using the hipotenusa between 2 datapoints
    Parameters
    ----------
    s: array-like
      the input signal.
    Returns
    -------
    signal distance: if the signal was straightened distance
    """
    df_sig = abs(np.diff(sig))
    return np.sum([np.sqrt(1 + df ** 2) for df in df_sig])


def deep_breath(sig):
    #the signal of the first apnea is received. The deep breath is characterized by a
    #sudden increase, which corresponds to a high value of the positive derivative.
    #to find the deep breath we use find extremes
    ind, peaks = find_extremes(sig, 'both', 0.3)
    assert(peaks != [])
    div1 = np.diff(peaks)
    deep_b = np.max(div1)
    if deep_b > 0:
        print('\ndeep breath was found')
        deep_idx = int(np.where(div1 == deep_b)[0])
        res = [peaks[deep_idx], peaks[deep_idx+1]]
    else:
        print(peaks, div1)
        res = 0
    print(res)
    return res


def load_bitalino(all_files, sr, win = 60, resample_len=1000, lab = 'bit', modality = 'resp'):
    x_train = np.array()
    Y_ = []
    # ONLY ONE FILE
    print('Files are loading ...\n')
    counter, rej = 0, 0
    for af in range(len(all_files)):
        print(all_files[af])
        each_train = []
        # df = pd.DataFrame.read_csv(all_files[af], sep=';', names=['DateTime', 'X'])
        # import pandas
        df = pd.read_csv(all_files[af], parse_dates=True, index_col=0, header=0, sep=';')
        # the vector is normalized
        X = bp.tools.filter_signal(df.A1, ftype='butter', band='bandpass', order=2, frequency=[0.1, 0.35], sampling_rate=sr)[
            'signal']

        deep = X[:]
        xminmax = deep_breath(deep)

        for k in range(0, len(X) - win, int(win)):
            each_train += [signal.resample(normal(X[k:k + win], xminmax[0], xminmax[1]), resample_len)]

        x_train.append(np.array(each_train))
    print('Finished')
    pickle.dump(np.array(x_train), open(lab + '_X', 'wb'))


def load_bit_rgbt(folder, all_files, sr, window=60, lab='bit', modality = 'resp', resample_len = 1000):
    """Load bitalino files from rgbt database


    
    """

    # annotations
    marker_label = dict([(2, 'Relax'), (5, 'Sinus'), (8, '1Ap'), (10, '2Ap'), (12, '3Ap'), (14, '4Ap'), (16, '5Ap')])

    X, Y = [], []

    window_pts = int(sr * window)

    print('Files are loading ...\n')
    counter, rej = 0, 0
    for af in range(len(all_files)):
        print(all_files[af])
        each_train = []
        each_y = []
        df = pd.read_csv(os.path.join(folder, all_files[af]), parse_dates=True, index_col=0, header=0, sep=';')
        if modality == 'resp':
            all_sig = bp.tools.filter_signal(df.A1, ftype='butter', band='bandpass', order=2, frequency=[0.1, 0.35], sampling_rate=sr)[
            'signal']
        elif modality == 'ppg':
            all_sig = bp.tools.filter_signal(df.A4, ftype='butter', band='bandpass',
                                  order=4, frequency=[1, 8], sampling_rate=sr)['signal']

        # the vector is segmented into samples of fixed size window = 2000
        # the samples are resized to a length of 1024 to fit in the autoencoder
        X_cut = [signal.resample(all_sig[i:i + window_pts], resample_len) for i in range(0, len(all_sig) - window_pts, window_pts)]
        # X_cut = [X[i:i + win] for i in range(0, len(X)-win, win)]
        # the first 20% goes to test and the rest goes to training
        list_markers = [[mk, marker] for mk, marker in enumerate(df.MARKER) if
                        marker != 0 and marker in marker_label.keys()]
        # before first apnea
        deep = all_sig[list_markers[3][0]:list_markers[4][0]]
        xminmax = deep_breath(deep)

        for j in range(0, len(list_markers), 2):
            # deep breath normalized
            sig = normal(all_sig[list_markers[j][0]:list_markers[j + 1][0]], xminmax[0], xminmax[1])
            # not normalized
            unsig = all_sig[list_markers[j][0]:list_markers[j + 1][0]]
            

            if 'Ap' in marker_label[list_markers[j][1]]:
                # remove first 10 seconds = 60s 
                new_sig = signal.resample(normal(unsig[10*sr:window_pts], xminmax[0], xminmax[1]), 1000).reshape(1, -1)

                each_y += ['A']
                        
            else:
                window_crop = window_pts - 10
                new_sig = []
                for i in range(0, len(unsig) - window_crop, window_crop):
                    new_sig += [signal.resample(normal(unsig[i:i+window_crop], xminmax[0], xminmax[1]), 1000) ]
                    each_y += ['N']
            if j == 0:
                user_x = np.vstack(new_sig)
            else:
                user_x = np.vstack((user_x, new_sig))

        X += [user_x]
        Y += [each_y]
        # print(each_y)

    print('Finished')
    pickle.dump(np.array(X), open(lab + '_X', 'wb'))
    pickle.dump(np.array(Y), open(lab + '_Y', 'wb'))


folder = "C:\\Users\\Mariana\\PycharmProjects\\ML_intro\\data\\APNEIA"

all_files = os.listdir(folder)


load_bit_rgbt(folder, all_files, sr=1000, lab='2022')