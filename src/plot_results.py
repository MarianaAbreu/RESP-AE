# This script is used to plot the results of the experiments. 
# It is used to plot the results of the experiments in the paper.
# The results are saved in the results folder.
# Author: Mariana Abreu
# Date: 2021


import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sn

# import data
user_labs = pickle.load(open('data/user_labs', 'rb'))
user_data = pickle.load(open('data/user_data', 'rb'))

# labels
labs = ['Free', 'Sinus', 'Apnea']


def normal(sg, minsg, maxsg):
    """
    Normalizes the signal
    :param sg: signal
    :param minsg: minimum value of the signal
    :param maxsg: maximum value of the signal
    :return res: normalized signal
    """

    res = 100 * (sg - minsg) / (maxsg - minsg)
    return res


def plot_rgbt_data():
    """
    Plots the signals from the BA-RGBT database
    Saves in results folder
    """
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
    fig.suptitle('Unscontrained (Free), Regular (Sinus) and Apnea samples from the BA-RGBT database')

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

    plt.savefig('results/signal_apnea.eps', bbox_inches='tight', format='eps')
    plt.show()

plot_rgbt_data()