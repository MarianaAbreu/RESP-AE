# ## Extract features from Respiration Signals
# 
# In this example we will follow the tutorial which is explained in detail here: https://blog.keras.io/building-autoencoders-in-keras.html
# 
# We will be running keras on tensorflow. To install both packages and dependencies, this video provides a very useful explanation https://www.youtube.com/watch?v=59duINoc8GM
# 

# ## 2. Load data
#

from scipy.stats import pearsonr
import time
import pylab as pl
from IPython import display

import pickle


#print(x_ap_train[0])
# df = pd.DataFrame.from_csv(all_files[6], parse_dates=True, index_col=0, header=0,sep=';')
# X = bp.tools.filter_signal(df.A1, ftype='butter', band='bandpass', order=2, frequency=[0.1, 1], sampling_rate=1000.0)['signal']
# X = normal(X)
# list_marker = [[mk,marker] for mk, marker in enumerate(df.MARKER) if marker != 0 and marker in marker_label.keys()]

resp = [bp.resp.resp(xap)['resp_rate'] for xap in x_ap_train]
# plt.figure(figsize=(30,10))
# plt.plot(normal(labels_test[200:800]))
# plt.plot(resp['ts'], resp['filtered'])
# plt.plot(resp['resp_rate_ts'], resp['resp_rate'])
# plt.ylim(-0.3,0.4)
# plt.show()


# In[143]:


resp.shape


# In[173]:





# In[146]:


#@title Try more classifiers
#@title Removal of highly correlated features

# Concatenation of entire data
features = pd.concat([X_train, X_test])
# Highly correlated features are removed
features = tsfel.correlation_report(features)
X_train = features[:len(X_train)]
X_test = features[len(X_train):]
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

y_train = np.array(labels_train)
y_test = np.array(labels_test)

labels = np.concatenate([y_train, y_test])
# Finds best supervised learning classifier
classifier = tsfel.find_best_slclassifier(features, labels, X_train, X_test, y_train, y_test)


# In[147]:



from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

#@title Feature Selection
# Feature Selection
FS_X_train, FS_X_test, FS_lab_description = tsfel.FSE(X_train, X_test, y_train, y_test, list(X_train.columns), classifier)
    
# Train the classifier
classifier.fit(FS_X_train, y_train.ravel())

# Predict test data
y_test_predict = classifier.predict(FS_X_test)
print('\n')

# Get the classification accuracy
accuracy = accuracy_score(y_test, y_test_predict)*100
scores = cross_val_score(classifier, np.concatenate([FS_X_train,FS_X_test]), labels, cv=10)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
print("Accuracy: " + str(accuracy) + '%')
print(classification_report(y_test, y_test_predict))


# In[64]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    # Only use the labels that appear in the data
    classes = ['Relax', 'Sinus', 'Apneia']

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# Plot non-normalized confusion matrix

plot_confusion_matrix(y_test, y_test_predict, classes=[],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
#plot_confusion_matrix(labels_test, pred3, classes=class_names, normalize=True,
 #                     title='Normalized confusion matrix')

plt.show()


# In[180]:


FS_X_train = [0]*10
FS_X_test = [0]*10
y_test = [0]*10
y_train = [0]*10
fold = 0
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, random_state=42)

labels_all = np.array(labels_all)
for train_index, test_index in kf.split(features_all):
    print(x_all_train.shape)
    print(test_index, train_index)
    FS_X_train[fold] = np.vstack(x_all_train[train_index])
    FS_X_test[fold] = np.vstack(x_all_train[test_index])
    y_train[fold] = np.concatenate(labels_all[train_index], axis=0)
    y_test[fold] = np.concatenate(labels_all[test_index], axis=0)
    fold +=1
FS_X_train = np.array(FS_X_train)
FS_X_test = np.array(FS_X_test)
y_test = np.array(y_test)
y_train = np.array(y_train)


# In[ ]:


FS_X_train = np.array(FS_X_train)
FS_X_test = np.array(FS_X_test)
y_test = np.array(y_test)
y_train = np.array(y_train)


# In[282]:


from scipy.stats import pearsonr
import time
import pylab as pl
from IPython import display

import pickle
decoder = pickle.load(open('resp_decoder', 'rb'))
encoder = pickle.load(open('resp_encoder', 'rb'))
print(type(x_apneia_train[0][0]))
print(np.array(x_apneia_train[0]).shape)

encoded_resp = encoder.predict(np.array(x_apneia_train[:]))
decoded_resp = decoder.predict(encoded_resp)
encoded_test = encoder.predict(np.array(x_apneia_test[:]))
decoded_test = decoder.predict(encoded_test)
#decoded_resp = decoder.predict(encoded_resp)
#x_train, x_test = load_timeseries(all_files, sampling_rate, window, div)
def compare(x_apneia_train, decoded_resp, labels_train):
    dst_all = []
    counter=0

    for i in range(len(x_apneia_train)):
        dst = pearsonr(decoded_resp[i], x_apneia_train[i])[0]
        dst_all +=[dst]
        if dst < 0.4:
            counter+=1
            plt.cla()
            plt.plot(x_apneia_train[i], label=(dst,i))

            plt.plot(decoded_resp[i], label='pred' )
            plt.legend()
            display.display(pl.gcf())
            display.clear_output(wait=True) 
            time.sleep(1.0)
    print(counter/len(x_apneia_train))
    plt.figure(figsize=(20,10))
    #plt.plot(dst_all)
    dst_smoo = bp.tools.smoother(dst_all, size=50)['signal']

    plt.plot(dst_smoo)
    print(set(np.diff(labels_train)))
    plt.ylim(0.9, 1.0)
    plt.vlines([i for i in range(len(labels_train)-1) if labels_train[i+1]-labels_train[i]==-4], 0, 1, colors='k', linestyles='solid')
    plt.vlines([i for i in range(len(labels_train)-1) if labels_train[i+1]-labels_train[i]==1], 0, 1, colors='g', linestyles='solid')
    plt.vlines([i for i in range(len(labels_train)-1) if labels_train[i+1]-labels_train[i]==3], 0, 1, colors='orange', linestyles='solid')


    pp = [bp.tools.pearson_correlation(decoded_resp[i],x_apneia_train[i])['rxy'] for i in range(len(x_apneia_train))]
    print(np.mean(pp),np.std(pp))

compare(x_apneia_train, decoded_resp, labels_train)
#compare(x_apneia_test, decoded_test, labels_test)


# In[224]:


from sklearn import tree
clf = tree.DecisionTreeClassifier()
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
neigh = KNeighborsClassifier(n_neighbors=5)
ssvm = svm.SVC(gamma='scale')
#normalize
import numpy as np

encoded_resp = encoder.predict(np.array(x_apneia_train[:]))
encoded_test = encoder.predict(np.array(x_apneia_test[:]))
clf = clf.fit(encoded_resp, labels_train)
pred = clf.predict(encoded_test)
nei = neigh.fit(encoded_resp, labels_train)
pred2 = nei.predict(encoded_test)
ssvm = ssvm.fit(encoded_resp, labels_train)
pred3 = ssvm.predict(encoded_test)
from sklearn.metrics import accuracy_score

ecc = accuracy_score(pred,labels_test)
ecc2 = accuracy_score(pred2,labels_test)
ecc3 = accuracy_score(pred3,labels_test)
print(ecc, ecc2, ecc3)


# In[62]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    # Only use the labels that appear in the data
    classes = ['Relax', 'Sinus', 'Apneia']

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix

plot_confusion_matrix(y_test, y_test_predict, classes=[],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
#plot_confusion_matrix(labels_test, pred3, classes=class_names, normalize=True,
 #                     title='Normalized confusion matrix')

plt.show()

