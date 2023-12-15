
import matplotlib.pyplot as plt
import seaborn as sb
import biosppy.signals as bp

import numpy as np
import pandas as pd
import os
import pickle
from scipy import interpolate

from scipy import signal
folder = "C:\\Users\\Mariana\\Documents\\Databases\\APNEIA\\"
# all_files = [folder+ol for ol in os.listdir(folder)]
all_files = [folder + ol for ol in os.listdir(folder)]
# samplint rate, window size of each sample
sampling_rate = 1000.
window = 10000
# percentage of the dataset that will be used for testing
div = 0.2

def pca():
    import pylab as plt
    import numpy as np
    import seaborn as sns;
    sns.set()

    import keras
    from keras.datasets import mnist
    from keras.models import Sequential, Model
    from keras.layers import Dense
    from keras.optimizers import Adam

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784) / 255
    x_test = x_test.reshape(10000, 784) / 255
    mu = x_train.mean(axis=0)
    U, s, V = np.linalg.svd(x_train - mu, full_matrices=False)
    Zpca = np.dot(x_train - mu, V.transpose())

    Rpca = np.dot(Zpca[:, :2], V[:2, :]) + mu  # reconstruction
    err = np.sum((x_train - Rpca) ** 2) / Rpca.shape[0] / Rpca.shape[1]
    print('PCA reconstruction error with 2 PCs: ' + str(round(err, 3)))

    m = Sequential()
    m.add(Dense(512, activation='elu', input_shape=(784,)))
    m.add(Dense(128, activation='elu'))
    m.add(Dense(2, activation='linear', name="bottleneck"))
    m.add(Dense(128, activation='elu'))
    m.add(Dense(512, activation='elu'))
    m.add(Dense(784, activation='sigmoid'))
    m.compile(loss='mean_squared_error', optimizer=Adam())
    history = m.fit(x_train, x_train, batch_size=128, epochs=1, verbose=1,
                    validation_data=(x_test, x_test))

    encoder = Model(m.input, m.get_layer('bottleneck').output)
    Zenc = encoder.predict(x_train)  # bottleneck representation
    Renc = m.predict(x_train)  # reconstruction

    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.title('PCA')
    plt.scatter(Zpca[:5000, 0], Zpca[:5000, 1], c=y_train[:5000], s=8, cmap='tab10')
    plt.gca().get_xaxis().set_ticklabels([])
    plt.gca().get_yaxis().set_ticklabels([])

    plt.subplot(122)
    plt.title('Autoencoder')
    plt.scatter(Zenc[:5000, 0], Zenc[:5000, 1], c=y_train[:5000], s=8, cmap='tab10')
    plt.gca().get_xaxis().set_ticklabels([])
    plt.gca().get_yaxis().set_ticklabels([])

    plt.tight_layout()

    plt.show()

def plot_result():
    import seaborn as sb
    sb.set(font_scale=1.5)
    acc = [[93.88, 93.02],
 [83.12, 76.36],
 [68.42, 53.85],
 [75.76, 57.89],
 [85.71, 88.89],
 [70.68, 29.09],
 [94.29, 93.94],
 [90.91, 90.0],
 [67.83, 17.78],
 [68.75, 80.77],
 [86.96, 89.66],
 [100.0, 100.0],
 [98.46, 98.51],
 [94.92, 94.74],
 [97.87, 97.96],
 [85.25, 80.85],
 [67.77, 29.09],
 [77.55, 74.42],
 [74.67, 48.65],
 [93.62, 92.68],
 [82.76, 73.68],
 [97.78, 97.87],
 [100.0, 100.0],
 [95.24, 95.65],
 [66.67, 45.71],
 [94.92, 95.38],
 [94.25, 93.51],
 [88.52, 86.27],
 [80.0, 85.71],
 [89.8, 87.18],
 [75.41, 51.61],
 [87.8, 87.8],
 [100.0, 100.0],
 [96.97, 97.14],
 [96.97, 97.14],
 [83.33, 87.5],
 [94.74, 95.24],
 [94.34, 94.92],
 [89.47, 87.76],
 [86.32, 81.16],
 [91.67, 91.18],
 [81.63, 70.97],
 [72.73, 51.61],
 [74.07, 46.15],
 [100.0, 100.0]
]
    ac = pd.DataFrame(acc,columns=['Apneia','Breathe'])
    fig = plt.figure(figsize=(10,2))
    plt.title('Results of F1_score for both classes')

    plt.vlines(np.arange(45), 0, 100, alpha=0.7, color='#a8a8a8')
    plt.scatter(np.arange(45),ac['Apneia'].values, label='Apnea')
    plt.scatter(np.arange(45), ac['Breathe'].values, label = 'Breathe')
    print(np.mean(ac['Apneia']))
    print(np.std(ac['Apneia']))
    print(np.mean(ac['Breathe']))
    print(np.std(ac['Breathe']))
    plt.xlabel('Users')
    plt.ylabel('F1_score (%'
               ''
               ')')
    plt.legend()

    plt.show()


plot_result()

def normal(sg, minsg, maxsg):
    res = 2 * (sg - minsg) / (maxsg - minsg) - 1
    return res

def deep_breath(sig):
    #the signal of the first apnea is received. The deep breath is characterized by a
    #sudden increase, which corresponds to a high value of the positive derivative.
    #to find the deep breathe we use find extremes
    ind, peaks = find_extremes(sig, 'both', 0.3)
    if peaks == []:
        plt.plot(sig)
        plt.scatter(ind,peaks)
        plt.show()
    div1 = np.diff(peaks)
    deep_b = np.max(div1)
    if deep_b > 0:
        print('\ndeep breathe was found')
        deep_idx = int(np.where(div1 == deep_b)[0])
        res = [peaks[deep_idx], peaks[deep_idx+1]]
    else:
        print(peaks, div1)
        res = 0
    print(res)
    return res
def find_extremes(sig, mode, th):
    indexes, values = bp.tools.find_extrema(sig, mode)
    ind, peaks = [], []
    for pk in range(len(values)):
        if abs(values[pk] - np.mean(sig)) > float(th):
            ind += [indexes[pk]]
            peaks += [values[pk]]
    return ind, peaks


def plot_ts(all_files, sr, win, div, lab):
    x_train, x_test = [], []
    Y_ = []
    # ONLY ONE FILE
    print('Files are loading ...\n')
    counter, rej = 0, 0
    for af in [9, 12, 40]:
        print(all_files[af])

        df = pd.DataFrame.from_csv(all_files[af], parse_dates=True, index_col=0, header=0, sep=';')
        # the vector is normalized
        X = \
        bp.tools.filter_signal(df.A1, ftype='butter', band='bandpass', order=2, frequency=[0.1, 0.4], sampling_rate=sr)[
            'signal']
        xn = normal(X,np.min(X), np.max(X))

        list_markers = [[mk, marker] for mk, marker in enumerate(df.MARKER) if
                        marker != 0 and marker in marker_label.keys()]
        markers = [lm[0] / 1000 for lm in list_markers]

        resp = bp.resp.ecg_derived_respiration(df.A2, xn,show=True)


        # ecg_ext = deep_breath(ecg_rate[find_nearest(ts_rate, list_markers[4][0])-5000:
        #                              find_nearest(ts_rate, list_markers[5][0])])

        """

        for j in range(0, len(list_markers), 2):

            sig = normal(X[list_markers[j][0]:list_markers[j + 1][0]], xminmax[0], xminmax[1])
            unsig = X[list_markers[j][0]:list_markers[j + 1][0]]

            # sig = normal(unsig, np.min(unsig), np.max(unsig))
            indexes, peaks = find_extremes(sig, 'both', 0.1)
            #plt.plot(sig, label=(marker_label[list_markers[j][1]], np.round(np.var(sig), 4)))
            win = 10000
            for k in range(0, len(sig) - win, int(win/2)):
                # print(j)
                ecg = ecg_rate[
                      find_nearest(ts_rate, list_markers[j][0]+k):find_nearest(ts_rate, list_markers[j][0]+k+win)]
                intp = int_sig[
                      find_nearest(ts_rate, list_markers[j][0]+k):find_nearest(ts_rate, list_markers[j][0]+k+win)]

                _peaks = [peaks[idx] for idx in range(len(indexes)) if k <= indexes[idx] <= k + win]
                _indexes = [idx for idx in indexes if k <= idx <= k + win]

                if _indexes:
                    result = dist(_peaks) * len(_peaks) * 2500 / (_indexes[-1] - _indexes[0])
                    # plt.scatter(_indexes,_peaks)
                else:
                    result = 'No peaks'
                print(result)

                if 'Ap' in marker_label[list_markers[j][1]]:
                    if str(result) == 'nan' or result == 'No peaks':
                        #each_train += [signal.resample(normal(unsig[k:k + win], xminmax[0], xminmax[1]), 1000)]
                        each_train += [signal.resample(intp, 1000)]
                        each_y += ['Apneia']
                        counter += 1
                    else:
                        rej += 1
                    # print(result,'Removed')
                elif 'Sinus' in marker_label[list_markers[j][1]]:
                #else:
                    if str(result) != 'nan' or result != 'No peaks':
                        # print(result,'Removed')
                        # else:
                        #each_train += [signal.resample(normal(unsig[k:k + win], xminmax[0], xminmax[1]), 1000)]
                        each_train += [signal.resample(intp, 1000)]
                        each_y += ['Breathe']
                        #counter+= 1
                        #each_y += [marker_label[list_markers[j][1]]]
                        # print(result,each_y[-1])
                    #else:
                     #   rej += 1

        x_train += [each_train]
        Y_ += [each_y]
        # print(each_y)

        """

    print(counter, rej)

    print('Finished')
    pickle.dump(np.array(x_train), open(lab + '_X', 'wb'))
    pickle.dump(np.array(Y_), open(lab + '_Y', 'wb'))


def find_nearest(array, value):

    return np.argmin(np.abs(array-value/1000))


marker_label = dict([(2, 'Relax'), (5, 'Sinus'), (8, 'Apnea'), (10, 'Apnea'), (12, 'Apnea'), (14, 'Apnea'), (16, 'Apnea')])

def load_timeseries2(all_files, sr, win, div, lab):
    x_train, x_test = [], []
    Y_ = []
    # ONLY ONE FILE
    print('Files are loading ...\n')
    counter, rej = 0, 0
    for af in [9,12,40]:
        print(all_files[af])
        each_train = []
        each_y = []
        # df = pd.DataFrame.read_csv(all_files[af], sep=';', names=['DateTime', 'X'])
        # import pandas
        df = pd.DataFrame.from_csv(all_files[af], parse_dates=True, index_col=0, header=0, sep=';')
        # the vector is normalized
        X = bp.tools.filter_signal(df.A1, ftype='butter', band='bandpass', order=2, frequency=[0.1, 0.4], sampling_rate=sr)[
            'signal']
        #ppg = bp.tools.filter_signal(df.A4, ftype='butter', band='bandpass',
        #                           order=4, frequency=[1, 8], sampling_rate=sr)['signal']
        # X = normal(np.array(X))

        # the vector is segmented into samples of fixed size 2000
        # the samples are resized to a length of 1024 to fit in the autoencoder
        # the first 20% goes to test and the rest goes to training
        list_markers = [[mk, marker] for mk, marker in enumerate(df.MARKER) if
                        marker != 0 and marker in marker_label.keys()]
        markers = [lm[0]/1000 for lm in list_markers]
        # print(list_markers)
        # print(df.columns)
        resp = bp.resp.ecg_derived_respiration(df.a2,X,show=True)
        ecg_ = bp.ecg.ecg(df.A2, show=False)
        bvp_ = bp.bvp.bvp(df.A4, show=False)
        bvp_hr = bvp_['heart_rate']
        bvp_hrts = bp.bvp.bvp(df.A4, show=False)['heart_rate_ts']
        eda_sig = bp.eda.eda(df.A3, show=False)['filtered']
        ecg_rate = normal(ecg_['heart_rate'],np.min(ecg_['heart_rate']),np.max(ecg_['heart_rate']))
        ecg_sig = ecg_['heart_rate']
        eee_filt = ecg_['filtered']
        ts_rate = ecg_['heart_rate_ts']
        idx_peaks = ecg_['rpeaks']
        print(idx_peaks)
        print(eee_filt)
        bvp_filt = normal(bvp_['filtered'], np.min(bvp_['filtered']),np.max(bvp_['filtered']))
        bvp_ipk, bvp_peaks = find_extremes(bvp_filt, mode='max',th=0.1)
        ecg_peaks = [eee_filt[e]  for e in range(len(eee_filt)-1) if e in idx_peaks]
        print(ecg_peaks)
        interp_bvp = interpolate.interp1d(bvp_ipk, bvp_peaks, kind='quadratic')

        deep = X
        xminmax = deep_breath(deep)
        #ecg_ext = deep_breath(ecg_rate[find_nearest(ts_rate, list_markers[4][0])-5000:
         #                              find_nearest(ts_rate, list_markers[5][0])])

        int_bvp = interp_bvp(np.arange(bvp_ipk[0],bvp_ipk[-1]))

        xii = np.concatenate([normal(X[xs:xs+10000],np.min(X[xs:xs+10000]),np.max(X[xs:xs+10000]))
                              for xs in range(0,len(X),10000)])

        """

        for j in range(0, len(list_markers), 2):

            sig = normal(X[list_markers[j][0]:list_markers[j + 1][0]], xminmax[0], xminmax[1])
            unsig = X[list_markers[j][0]:list_markers[j + 1][0]]

            # sig = normal(unsig, np.min(unsig), np.max(unsig))
            indexes, peaks = find_extremes(sig, 'both', 0.1)
            #plt.plot(sig, label=(marker_label[list_markers[j][1]], np.round(np.var(sig), 4)))
            win = 10000
            for k in range(0, len(sig) - win, int(win/2)):
                # print(j)
                ecg = ecg_rate[
                      find_nearest(ts_rate, list_markers[j][0]+k):find_nearest(ts_rate, list_markers[j][0]+k+win)]
                intp = int_sig[
                      find_nearest(ts_rate, list_markers[j][0]+k):find_nearest(ts_rate, list_markers[j][0]+k+win)]

                _peaks = [peaks[idx] for idx in range(len(indexes)) if k <= indexes[idx] <= k + win]
                _indexes = [idx for idx in indexes if k <= idx <= k + win]

                if _indexes:
                    result = dist(_peaks) * len(_peaks) * 2500 / (_indexes[-1] - _indexes[0])
                    # plt.scatter(_indexes,_peaks)
                else:
                    result = 'No peaks'
                print(result)

                if 'Ap' in marker_label[list_markers[j][1]]:
                    if str(result) == 'nan' or result == 'No peaks':
                        #each_train += [signal.resample(normal(unsig[k:k + win], xminmax[0], xminmax[1]), 1000)]
                        each_train += [signal.resample(intp, 1000)]
                        each_y += ['Apneia']
                        counter += 1
                    else:
                        rej += 1
                    # print(result,'Removed')
                elif 'Sinus' in marker_label[list_markers[j][1]]:
                #else:
                    if str(result) != 'nan' or result != 'No peaks':
                        # print(result,'Removed')
                        # else:
                        #each_train += [signal.resample(normal(unsig[k:k + win], xminmax[0], xminmax[1]), 1000)]
                        each_train += [signal.resample(intp, 1000)]
                        each_y += ['Breathe']
                        #counter+= 1
                        #each_y += [marker_label[list_markers[j][1]]]
                        # print(result,each_y[-1])
                    #else:
                     #   rej += 1

        x_train += [each_train]
        Y_ += [each_y]
        # print(each_y)
        
        """
        #sb.set(font_scale=2)
        plt.figure(figsize=(20,10))
        #plt.subplot(3, 1, 1)
        #X=X[list_markers[5][0]:]
        xax = np.arange(0,len(X)/1000,0.001)

        #plt.title('Heart Rate')
        #plt.plot(ts_rate, normal(ecg_rate, np.min(ecg_rate),np.max(ecg_rate)))
        #int_sig = interp(np.arange(idx_peaks[0],idx_peaks[-1]))
        #plt.title('R Peaks Interpolation')

        #plt.plot(ts_rate, signal.resample(normal(ecg_rate, np.min(ecg_rate), np.max(ecg_rate)),len(ecg_rate)), label='HRV')
        #plt.plot(ts_rate, signal.resample(normal(int_sig, np.min(int_sig), np.max(int_sig)), len(ecg_rate)), label ='R Interpolation')
        #plt.plot(ts_rate, signal.resample(normal(X, np.min(X), np.max(X)), len(ecg_rate)),label='Respiration')

        plt.subplot(4,1,1)
        plt.ylim(-1.4, 1.4)
        #plt.vlines(markers, -1.4, 1.4, linewidth=2, zorder=10)
        #for lm in range(0, len(list_markers), 2):
         #   plt.text(((list_markers[lm + 1][0] - list_markers[lm][0]) / 2 + list_markers[lm][0]) / 1000,
          #           1.05, marker_label[list_markers[lm][1]].upper(), horizontalalignment='center', fontsize=15)


        plt.title('Respiratory Signal')
        plt.plot(xax, (normal(X, xminmax[0], xminmax[1])),
                 linewidth=2,color='#00304d',zorder=8, alpha=0.8)
        #plt.plot(np.arange(0,len(int_sig)/1000,0.001), (normal(int_sig, np.min(int_sig),np.max(int_sig))), linewidth=2, color='#006eb3', zorder=2, alpha=0.8)
        plt.xticks([])
        plt.subplot(4, 1, 2)
        plt.ylim(-1.4, 1.4)
        #plt.vlines(markers, -1.4, 1.4, linewidth=2, zorder=10)

        plt.xticks([])
        plt.title('Heart Rate Variability from ECG')
        plt.ylabel('Amplitude', fontsize=25)
        plt.plot(ts_rate, (normal(ecg_sig,np.min(ecg_sig),np.max(ecg_sig))), linewidth=2,color='#006eb3',zorder=2, alpha=0.8)
        #plt.plot(xax, (normal(xii, np.min(xii), np.max(xii))),
         #        linewidth=2,color='#006eb3',zorder=2, alpha=0.8)
        #plt.plot(xax, bvp_filt)
        plt.subplot(4, 1, 3)
        plt.ylim(-1.4, 1.4)
        #plt.vlines(markers, -1.4, 1.4, linewidth=2, zorder=10)
        plt.xticks([])

        plt.title('Heart Rate Variability from BVP')
        #plt.plot(xax, (normal(X, xminmax[0], xminmax[1])),
         #       linewidth=2, color='#0081cc', zorder=2, alpha=0.8)
        plt.plot(bvp_hrts, (normal(bvp_hr, np.min(bvp_hr), np.max(bvp_hr))), linewidth=2, color='#006eb3', zorder=2,
                 alpha=0.8)
        #plt.plot(np.arange(0, len(X) / 1000, 0.001), (normal(X, xminmax[0], xminmax[1])), linewidth=2)
        plt.subplot(4, 1, 4)
        plt.ylim(-1.4, 1.4)
        #plt.vlines(markers, -1.4, 1.4, linewidth=2, zorder=10)

        plt.title('Interpolation from BVP')
        #plt.plot(xax, (normal(eda_sig, np.min(eda_sig), np.max(eda_sig))), linewidth=2, color='#006eb3', zorder=2,
         #        alpha=0.8)
        plt.plot(np.arange(0,len(int_bvp)/1000,0.001), (normal(int_bvp, np.min(int_bvp),np.max(int_bvp))), linewidth=2, color='#006eb3', zorder=2, alpha=0.8)
        #plt.plot(xax, (normal(X, xminmax[0], xminmax[1])),
         #        linewidth=2, color='#0081cc', zorder=2, alpha=0.8)

        plt.xlabel('Time (s)',fontsize=25)

        plt.legend()

        #plt.subplot(3,1,2)
        #plt.plot(bvp_ts, normal(bvp_rate, np.min(bvp_rate),np.max(bvp_rate)))

        #plt.plot(ts_rate, signal.resample(normal(int_sig,np.min(int_sig),np.max(int_sig)),len(ecg_rate)))
        #plt.plot(ts_rate,signal.resample(normal(bvp_rate,np.min(bvp_rate),np.max(bvp_rate)),len(ecg_rate)))
        #plt.plot(ts_rate, signal.resample(normal(eda_, np.min(eda_), np.max(eda_)), len(ecg_rate)))
        #plt.vlines(markers, -1, 1)
        #plt.subplot(3,1,3)
        plt.ylim(-1.4, 1.4)
        plt.hspace=1.5
        #plt.title('Respiration Signal', fontsize=30)
        plt.savefig('Bit' + str(af) +'.png', bbox_inches='tight', format='png')

        #plt.plot(ts_rate, signal.resample(normal(int_sig, np.min(int_sig), np.max(int_sig)), len(ecg_rate)))

        #plt.show()
    print(counter, rej)

    print('Finished')
    pickle.dump(np.array(x_train), open(lab + '_X', 'wb'))
    pickle.dump(np.array(Y_), open(lab + '_Y', 'wb'))

#plot_ts(all_files, sampling_rate, window, div, lab='intp')