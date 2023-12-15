from scipy.stats import pearsonr
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import pandas as pd
# names = ["Decision Tree", "AdaBoost","GradBoost"]
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# classifiers = [DecisionTreeClassifier(max_depth=5),
#     AdaBoostClassifier(),GradientBoostingClassifier()]
# names = ["GradBoost"]
# classifiers = [GradientBoostingClassifier()]

import random
from sklearn.model_selection import train_test_split

import pickle

def part_balance(list_vecs, list_lab):
    train = []
    lab = []
    for v, vec in enumerate(list_vecs):

        num = list_lab[v].count('A')
        if num != 0:
            sel = random.choices(vec[:-num], k=num)
            tr = np.concatenate([sel, vec[-num:]])
            ba = ['N'] * num + ['A'] * num

            train += [tr]
            lab += [ba]

            if len(tr) != len(ba):
                aaaa
    return train, lab


def autoencoder_transfer(model, encoding_dim_,input_len, list_layers, a_fun, optimizer, loss
                       ,x_train,y_train,x_test,y_test,label, epochs):
    encoding_dim = encoding_dim_  # 32 floats -> compression of factor 32, assuming the input is 1024 floats

    if sorted(list_layers, key=int, reverse=True) != list_layers:
        print('\nList should be in order!\n')
        popo
    input_sig = Input(shape=(input_len,))
    encoded = Dense(list_layers[0], activation=a_fun)(input_sig)
    network = list_layers + list_layers[::-1][1:] + [input_len]
    for nt in list_layers[1:]:
        encoded = Dense(nt, activation=a_fun)(encoded)
    # "decoded" is the lossy reconstruction of the input
    inverse_layers = list_layers[::-1][1:] + [input_len]
    decoded = Dense(inverse_layers[0], activation=a_fun)(encoded)
    for ll in inverse_layers[1:]:
        decoded = Dense(ll, activation=a_fun)(decoded)

    autoencoder = model
    ##Let's also create a separate encoder model as well as the decoder model:
    encoder = Model(input_sig, encoded)
    encoded_input = Input(shape=(encoding_dim,))

    dec_layers = autoencoder.layers[len(list_layers) + 1:]
    decoder_output = dec_layers[0](encoded_input)
    for dl in dec_layers[1:]:
        decoder_output = dl(decoder_output)
    decoder = Model(encoded_input, decoder_output)

    ##First, we'll configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
    autoencoder.compile(optimizer=optimizer, loss=loss)

    print('\nAutoencoder Created:')
    print('Layers: ' + str(list_layers))
    print('Input Length: ' + str(input_len))
    print('Compression: ' + str(encoding_dim))
    print('Activation: ' + str(a_fun))
    print('Optimizer: ' + str(optimizer))
    print('Loss: ' + str(loss) + '\n')

    autoencoder.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=100,
                    shuffle=True,
                    validation_data=(x_test, y_test))


    pickle.dump(decoder, open(label + '_decoder', 'wb'))
    pickle.dump(encoder, open(label + '_encoder', 'wb'))
    return encoder, decoder

def autoencoder_params(encoding_dim_,input_len, list_layers, a_fun, optimizer, loss
                       ,x_train,y_train,x_test,y_test,label, epochs):
    # this is the size of our encoded representations
    encoding_dim = encoding_dim_  # 32 floats -> compression of factor 32, assuming the input is 1024 floats

    if sorted(list_layers, key=int, reverse=True) != list_layers:
        print('\nList should be in order!\n')
        popo
    input_sig = Input(shape=(input_len,))
    encoded = Dense(list_layers[0], activation = a_fun)(input_sig)
    network = list_layers + list_layers[::-1][1:] + [input_len]
    for nt in list_layers[1:]:
        encoded = Dense(nt, activation=a_fun)(encoded)
    # "decoded" is the lossy reconstruction of the input
    inverse_layers = list_layers[::-1][1:] + [input_len]
    decoded = Dense(inverse_layers[0], activation=a_fun)(encoded)
    for ll in inverse_layers[1:]:
        decoded = Dense(ll, activation=a_fun)(decoded)

    autoencoder = Model(input_sig, decoded)
    ##Let's also create a separate encoder model as well as the decoder model:
    encoder = Model(input_sig, encoded)
    encoded_input = Input(shape=(encoding_dim,))

    dec_layers = autoencoder.layers[len(list_layers)+1:]
    decoder_output = dec_layers[0](encoded_input)
    for dl in dec_layers[1:]:
        decoder_output = dl(decoder_output)
    #decoder = Model(encoded_input, decoder_layer1(decoder_layer2(decoder_layer3(decoder_layer4(decoder_layer5(encoded_input))))))
    decoder = Model(encoded_input,decoder_output)

    ##First, we'll configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
    autoencoder.compile(optimizer=optimizer,loss=loss)

    print('\nAutoencoder Created:')
    print('Layers: ' + str(list_layers))
    print('Input Length: ' + str(input_len))
    print('Compression: ' + str(encoding_dim))
    print('Activation: ' + str(a_fun))
    print('Optimizer: ' + str(optimizer))
    print('Loss: ' + str(loss) +'\n')

    autoencoder.fit(x_train,y_train,
                    epochs=epochs,
                    batch_size=100,
                    shuffle=True,
                    validation_data=(x_test, y_test))


    pickle.dump(autoencoder, open(label + '_autoencoder', 'wb'))
    pickle.dump(decoder, open(label + '_decoder', 'wb'))
    pickle.dump(encoder, open(label + '_encoder', 'wb'))

    return autoencoder, encoder, decoder


def create_autoencoder(x_train, y_train, x_test, y_test, label, loss,
                       activ='tanh', opt='adam', nodes=[500, 250, 50], epochs=100):
    autoencoder,encoder,decoder = autoencoder_params(nodes[-1], 1000, nodes, activ, opt, loss,
                          x_train, y_train, x_test, y_test, label, epochs)
    return autoencoder, encoder, decoder

def separate_classes(vecs, labs, classes=['A', 'N']):
    # take the whole set and separate classes.
    new_vec, new_labs = [], []
    for cl in classes:
        cl_index = [i for i in range(len(labs)) if labs[i] == cl]
        vec_ = np.array(vecs)[cl_index]
        lab_ = np.array(labs)[cl_index]
        new_vec += [vec_]
        new_labs += [lab_]
    return new_vec, new_labs

names = ["KNN", "SVM", "Decision Tree", "Random Forest", "Neural Net", \
         "Adaboost", "GradBoost", "Naive Bayes", "QDA"]
"""
classifiers = [
    KNeighborsClassifier(3),
    SVC(gamma=2, C=1, probability=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

"""
# names = ["SVM", "Random Forest", "NeuralNet"]

def choose_classifier(data, labels, classifiers):
    best = 0
    cl_train, cl_test, cl_ytrain, cl_ytest = \
        train_test_split(data, labels, test_size=.4, random_state=42)
    all_scores = []
    for n, c in zip(names, classifiers):
        print(n)
        scores = cross_val_score(c, data, labels, scoring='f1', cv=10)
        print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

        # Train the classifier
        c.fit(cl_train, cl_ytrain.ravel())

        # Predict test data
        y_test_predict = c.predict(cl_test)

        # Get the classification accuracy
        print("Accuracy: " + str(scores.mean()) + '%')
        print('-----------------------------------------')
        if scores.mean() > best:
            best_classifier = n
            best = scores.mean()
        all_scores += [np.round(scores.mean() * 100, 2)]

    print('******** Best Classifier: ' + str(best_classifier) + ' ********')
    print(all_scores)
    return pd.DataFrame([all_scores], columns=names)


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
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

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

def score_classifier(c, n, train, test, y_train, y_test, show=False):
    all_scores = []

    # Train the classifier

    c.fit(train, y_train.ravel())

    # Predict test data
    y_pred = c.predict(test)

    #class_names = ['S', 'F', 'A']
    class_names = ['A', 'N']

    y_test = [class_names.index(yt) for yt in y_test]
    y_pred = [class_names.index(yp) for yp in y_pred]

    #if len(np.unique(y_test)) == 1:
     #   score = f1_score(y_test, y_pred, pos_label=y_test[0],average='binary')
    #else:
    #score_f1 = f1_score(y_test, y_pred, average='weighted')
    #score_f1 = f1_score(y_test, y_pred)
    score_f1 = 0
    score_acc = accuracy_score(y_test, y_pred)

    # Get the classification accuracy
    print(str(n) + " --- Accuracy: " + str(score_acc) + '%')
    print(str(n) + " --- F1 Score: " + str(score_f1) + '%')
    print('-----------------------------------------')
    #class_names = np.array(['Apnea', 'Normal'])
    #cm = confusion_matrix(y_test, y_pred)
    #print(cm)
    if show:
        # Plot non-normalized confusion matrix
        plot_confusion_matrix(y_test, y_pred, classes=class_names,
                              title='Confusion matrix, without normalization')
        plt.show()

    # return [np.round(score*100,2), np.round(accuracy*100,2)]
    return np.round(score_acc * 100, 2), np.round(score_f1 * 100, 2)
    #return y_pred, y_test


def fus_classifier(cl, ns, train, test, y_train, y_test, show=False):
    all_scores = []

    # Train the classifier
    y_pred = []

    cl[0].fit(train, y_train.ravel())
    cl[1].fit(train, y_train.ravel())
    proba1 = cl[0].predict_proba(test)
    proba2 = cl[1].predict_proba(test)
    # prob = [proba1[pb] if np.argmax([proba1[pb], proba2[pb]]) in [0,1] else proba2[pb] for pb in range(len(proba1))]
    prob = [[np.mean([proba1[i, 0], proba2[i, 0]]), np.mean([proba1[i, 1], proba2[i, 1]])] for i in range(len(proba1))]

    print(prob)
    y_pred = [np.argmax(prob[i]) for i in range(len(prob))]
    y_test = [1 if yt == 'N' else 0 for yt in y_test]
    print(y_test, y_pred)
    score = accuracy_score(y_test, y_pred)

    # print(c.predict_proba(test))
    # Get the classification accuracy
    print(str(ns) + " --- Accuracy: " + str(score) + '%')
    print('-----------------------------------------')
    class_names = np.array(['Apnea', 'Normal'])
    if show:
        # Plot non-normalized confusion matrix
        plot_confusion_matrix(y_test, y_pred, classes=class_names,
                              title='Confusion matrix, without normalization')

        plt.show()

    # return [np.round(score*100,2), np.round(accuracy*100,2)]
    return np.round(score * 100, 2)


def fusion_classifier(classifiers, names, train, test, y_train, y_test, show=False):
    all_scores = []

    cl_score = []

    for n, c in zip(names, classifiers):

        c.fit(train, y_train.ravel())

        prob = c.predict_proba(test)

        if n == names[0]:
            predictions = prob[0]
        else:
            predictions = np.column_stack(
                (np.maximum(predictions[0][:, 0], prob[0][:, 0]), np.maximum(predictions[0][:, 1], prob[0][:, 1])))

    y_pred = [np.argmax(np.array(yi)) for yi in predictions]

    print(y_pred)

    score = f1_score(y_test, y_pred, average='micro')

    # accuracy = accuracy_score(y_test,y_pred)
    # Get the classification accuracy
    print(" Fusion --- F1 Score: " + str(score) + '%')
    print('-----------------------------------------')
    class_names = np.array(['Apnea', 'Normal'])

    if show:
        # Plot non-normalized confusion matrix
        plot_confusion_matrix(y_test, y_pred, classes=class_names,
                              title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                              title='Normalized confusion matrix')

        plt.show()

        # return [np.round(score*100,2), np.round(accuracy*100,2)]
    return np.round(score * 100, 2)


def classification(training_set_x, testing_set_x, training_set_y, testing_set_y):
    train, y_train = separate_classes(training_set_x, training_set_y)

    n_train, n_test, n_ytrain, n_ytest = \
        train_test_split(train[1], y_train[1], test_size=.4, random_state=42)

    loss = 'cosine_proximity'
    activ = 'tanh'
    opt = 'adam'

    nodes = [500, 250, 50]

    create_autoencoder(n_train, n_train, n_test, n_test, 'RN', loss, activ, opt, nodes=nodes, epochs=50)

    encoder = pickle.load(open('RN_encoder', 'rb'))
    decoder = pickle.load(open('RN_decoder', 'rb'))

    enc = encoder.predict(testing_set_x)
    dec = decoder.predict(enc)

    enc_train = encoder.predict(training_set_x)
    dec_train = decoder.predict(enc_train)
    train_cl, test_cl = [], []
    y_train_new = []
    y_test_new = []
    for d in range(len(dec_train)):
        corr_train = [pearsonr(dec_train[d][i:i + 100], training_set_x[d][i:i + 100])[0] for i in
                      range(0, len(dec_train[d]), 100)]
        if np.isfinite(corr_train).all():
            train_cl += [corr_train]
            y_train_new += [training_set_y[d]]

    for d in range(len(dec)):

        corr = [pearsonr(dec[d][i:i + 100], testing_set_x[d][i:i + 100])[0] for i in range(0, len(dec[d]), 100)]

        if np.isfinite(corr).all():
            test_cl += [corr]
            y_test_new += [testing_set_y[d]]

    names = ["SVM", "RF", "NB"]

    classifiers = [SVC(gamma=2, C=1, probability=True),
                   RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), GaussianNB()]

    _score = fus_classifier(classifiers, names, train_cl, \
                            test_cl, np.array(y_train_new), np.array(y_test_new), show=False)
    return _score

