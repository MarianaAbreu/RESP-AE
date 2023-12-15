from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from scipy.stats import randint as sp_randint
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

FS_X_train = pickle.load(open("FS_X_train.pickle",'rb'))
FS_X_test = pickle.load(open("FS_X_test.pickle",'rb'))
FS_D_train = pickle.load(open("FS_D_train.pickle",'rb'))
FS_D_test = pickle.load(open("FS_D_test.pickle",'rb'))
Y_train_ = pickle.load(open("y_train.pickle",'rb'))
Y_test_ = pickle.load(open("y_test.pickle",'rb'))
FS_X = pickle.load(open("FS_X.pickle",'rb'))
FS_E_train = pickle.load(open('FS_E_train.pickle','rb'))
FS_E_test = pickle.load(open('FS_E_test.pickle','rb'))

#third = pickle.load(open("y_train3.pickle",'rb'))

labels = np.concatenate([Y_train_[0], Y_test_[0]])
features = np.concatenate(FS_X)
all_accuracy, all_scores = [],[]


def forward_selection_user(FS_X_train, FS_X_test, Y_train_, Y_test_, classifier):
    """

     :param features_labels: all features extracted
     :param X: shape(n_samples, n_features)
     :return:
     """
    # start with one feature
    features_labels = list(pd.read_csv('Features.csv').columns)[1:]
    #features_labels = [str(k) for k in range(32)]
    current_accuracy, last_accuracy, max_index = 0, -1, -1
    best_features = []
    index_x, first_bf = [], []
    max_feature = ''
    max_index = -1
    print('\n')
    print('****************************FORWARD FEATURE SELECTION*****************************')
    print('\n')

    cycle = 0

    while current_accuracy > last_accuracy:
        cycle += 1
        last_accuracy = current_accuracy  # update last accuracy
        print('Last Accuracy = ' + str(last_accuracy))
        accuracies_feature = []

        if max_index != -1:
            best_features.append(max_feature[1])

        feat_test_activity, feat_pred_activity = [], []


        for fl, which_feature in enumerate(features_labels):  # run all features still up for evaluation
            if which_feature not in best_features: # and 'total_energy' not in which_feature:

                print('-----------Feature used ' + str(which_feature) +
                      ' with best features ' + str(" ".join(best_features)) + '-------------' + '\n')
                print('CYCLE -----------' + str(cycle))
                users_test_activity, users_pred_activity, accuracies_list = [], [], []
                counter = 0

                while counter < len(FS_X_train):  # cross_validation
                    All_train = FS_X_train[counter]
                    All_test = FS_X_test[counter]
                    new_x_train = All_train[:,fl]
                    new_x_test = All_test[:,fl]
                    bf_x_test = np.array([All_test[:,k] for k in range(len(features_labels)) if features_labels[k] in best_features]).T
                    bf_x_train = np.array([All_train[:,k] for k in range(len(features_labels)) if features_labels[k] in best_features]).T

                    x_train = np.column_stack((bf_x_train, new_x_train)) if max_index != -1 else new_x_train   #choose only one activity
                    x_test = np.column_stack((bf_x_test, new_x_test)) if max_index != -1 else new_x_test   #choose only one activity


                    #print('X train shape -- ' + str(x_train.shape))
                    #print('X test shape -- ' + str(x_test.shape))

                        # single feature reshape data
                    if np.array([x_train[0]]).shape == (1,):
                        x_train = np.array(x_train).reshape(-1, 1)
                        x_test = np.array(x_test).reshape(-1, 1)

                        len_train_all = []

                    classifier.fit(x_train, Y_train_[counter].ravel())
                    y_pred = classifier.predict(x_test)
                    accuracy = accuracy_score(Y_test_[counter], y_pred) * 100
                    accuracies_list.append(accuracy)
                    counter += 1

                accuracies_feature.append((np.mean(accuracies_list), which_feature))
                print("Accuracy: " + str(np.mean(accuracies_list)) + '%' + '\n' + '\n')
                print(x_test.shape)

        print('Accuracies Feature List -- ' + str(accuracies_feature))
        max_accuracy = max(accuracies_feature)
        current_accuracy = float(max_accuracy[0])
        acur_index = accuracies_feature.index(max_accuracy)
        max_feature = accuracies_feature[acur_index]
        max_index = features_labels.index(max_feature[1])
        index_x.append(max_index)

        print("Feature " + str(max_feature) + " had best accuracy: " + str(max_accuracy) + '%')
    print('Best combination of features is -- ' + str(best_features) + ' for classifier Dec Tree' + ' with data E')
    print('Final accuracy is -- ' + str(last_accuracy) + '%')

    with open('best_features.pickle', 'wb') as handle:
        pickle.dump(best_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

def grid_trial_RF(X,Y):

    clf = RandomForestClassifier()

    X_train, X_test, y_train, y_test = train_test_split(X,Y,
                                                        test_size=0.33,random_state=42)
    params = {'n_estimators': [30, 50, 100],
              'max_features': ['sqrt', 'log2', 10, 30, 40]}
    print(' ----- GridSearch started  ----- ')
    gsv = GridSearchCV(clf, params, cv=3, n_jobs=-1, scoring='accuracy')
    gsv.fit(X_train, y_train)

    print(classification_report(y_train, gsv.best_estimator_.predict(X_train)))

    print(classification_report(y_test, gsv.best_estimator_.predict(X_test)))

    return gsv.best_estimator_

def find_cl(X,Y):
    print(Y)

    X_train, X_test, y_train, y_test = train_test_split(X,Y,
                                                        test_size=0.33,random_state=42)

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="rbf", C=0.025, probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()]
    all_accuracies = []

    # Logging for Visual Comparison
    log_cols = ["Classifier", "Accuracy", "Log Loss"]
    log = pd.DataFrame(columns=log_cols)

    for clf in classifiers:
        clf.fit(X_train, y_train)
        name = clf.__class__.__name__

        print("=" * 30)
        print(name)

        print('****Results****')
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        all_accuracies += [acc]
        print("Accuracy: {:.4%}".format(acc))

        train_predictions = clf.predict_proba(X_test)
        ll = log_loss(y_test, train_predictions)
        print("Log Loss: {}".format(ll))

        log_entry = pd.DataFrame([[name, acc * 100, ll]], columns=log_cols)
        log = log.append(log_entry)

    print(classifiers[np.argmax(all_accuracies)])
    print("=" * 30)

    # Predict Test Set
    favorite_clf = GradientBoostingClassifier()
    favorite_clf.fit(X_train, y_train)
    test_predictions = favorite_clf.predict_proba(X_test)

    # Format DataFrame
    submission = pd.DataFrame(test_predictions, columns=['Breathe', 'Apneia'])
    #submission.insert('value', 'id')
    submission.reset_index()

    # Export Submission
    submission.to_csv('submission.csv', index = False)
    submission.tail()

    return classifiers[np.argmax(all_accuracies)]


def GradBoost_cl_bin(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X,Y,
                                                        test_size=0.33,random_state=42)

    print("=" * 30)
    # Predict Test Set
    favorite_clf = GradientBoostingClassifier()
    favorite_clf.fit(X_train, y_train)
    level2 = []

    test_predictions = favorite_clf.predict(X_test)
    y_test_num = [1 if yt == 'Breathe' else 2 if yt == 'Apneia' else 0 for yt in y_test]
    pred_num = [1 if pt == 'Breathe' else 2 if pt == 'Apneia' else 0 for pt in test_predictions]


    acc = accuracy_score(y_test_num, pred_num)

    print(classification_report(y_test,test_predictions))

    print("Accuracy: {:.4%}".format(acc))
    print('+' * 20)

    return acc, len(X_test)


def GradBoost_cl(X,Y, second):
    X2_train = [X[th] for th in range(len(second)) if second[th] != 'Apneia']
    y2_train = [second[th] for th in range(len(second)) if second[th] != 'Apneia']

    X_train, X_test, y_train, y_test = train_test_split(X,second,
                                                        test_size=0.33,random_state=42)
    first_y = ['Breathe' if yt != 'Apneia' else yt for yt in y_train]
    first_y_test = ['Breathe' if yt != 'Apneia' else yt for yt in y_test]

    print("=" * 30)
    # Predict Test Set
    favorite_clf = GradientBoostingClassifier()
    favorite_clf.fit(X_train, first_y)
    breathe_clf = GradientBoostingClassifier()
    breathe_clf.fit(X2_train,y2_train)
    level2 = []

    test_predictions = favorite_clf.predict(X_test)
    predictions = []
    for tep in range(len(test_predictions)):

        if test_predictions[tep] == 'Breathe':
            level2 += [X_test[tep]]
            level2_pred = breathe_clf.predict([X_test[tep]])[0]
            level2_prob = breathe_clf.predict_proba([X_test[tep]])
            predictions.append(level2_pred)
        else:
            predictions.append(test_predictions[tep])

    prob_pred = favorite_clf.predict_proba(X_test)

    acc = accuracy_score(first_y_test, test_predictions)

    print("Accuracy: {:.4%}".format(acc))
    print('+' * 20)
    acc = accuracy_score(y_test, predictions)
    print("Accuracy: {:.4%}".format(acc))

    return acc, len(X_test)

def grid_trial_DT(X,Y):

    clf = DecisionTreeClassifier()

    X_train, X_test, y_train, y_test = train_test_split(X,Y,
                                                        test_size=0.33,random_state=42)
    params = {
              'max_features': ['sqrt', 'log2', 10, 30, 40]}
    print(' ----- GridSearch started  ----- ')
    gsv = GridSearchCV(clf, params, cv=3, n_jobs=-1, scoring='accuracy')
    gsv.fit(X_train, y_train)

    print(classification_report(y_train, gsv.best_estimator_.predict(X_train)))

    print(classification_report(y_test, gsv.best_estimator_.predict(X_test)))

    return gsv.best_estimator_


def grid_trial_SVM(X, y):


    #print(__doc__)

    # Split the dataset in two equal parts

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)
    print(set(y_train) - set(y_test))

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
    return clf, clf.best_params_


def hierachical(FS_X_train,Y_train_,third):
    all_acc = []
    all_le = 0
    for fsx in range(len(FS_X_train)):
        print(np.array(third[fsx]).shape)

        print(np.array(Y_train_[fsx]).shape)
        acc_gb, le = GradBoost_cl(FS_X_train[fsx], Y_train_[fsx], third[fsx])
        all_acc += [acc_gb*le]
        all_le += le
        print('final accuracy ---- ' + str(np.sum(all_acc)/all_le))

def regular(FS_X_train, Y_train_):
    all_acc = []
    all_le = 0
    for fsx in range(len(FS_X_train)):
        print(np.array(Y_train_[fsx]).shape)
        acc_gb, le = GradBoost_cl_bin(FS_X_train[fsx], Y_train_[fsx])
        all_acc += [acc_gb*le]
        all_le += le
    print('final accuracy ---- ' + str(np.sum(all_acc)*100/all_le))


#regular(FS_X_train, Y_train_)
#hierachical(FS_X_train,Y_train_,third)


def best_classifier():
    all_classifiers = []
    for fs in range(len(FS_X_train)):
        classifier = tsfel.find_best_slclassifier(features, labels, FS_X_train[fs], FS_X_test[fs], Y_train_[fs], Y_test_[fs])
        cl = LinearDiscriminantAnalysis()
        #print(type(classifier))
        print('Fold ' + str(fs) + ' chooses Classifier -- ' + str(classifier))
    print(set(all_classifiers))

    forward_selection_user(FS_X_train, FS_X_test, Y_train_, Y_test_, cl)



def last():
    for xt in range(len(FS_X_train)):
        X_train = pd.DataFrame(FS_X_train[xt])
        X_test = pd.DataFrame(FS_X_test[xt])
        Y_train = Y_train_[xt]
        Y_test = Y_test_[xt]
        #classifier = DecisionTreeClassifier(max_depth=5, min_samples_split=len(features)//10)
        #classifier = RandomForestClassifier(max_depth=5, n_estimators=10)
        classifier = LinearDiscriminantAnalysis()
        classifier.fit(X_train, Y_train.ravel())

        FS_new_train, FS_new_test, FS_lab_description = tsfel.FSE(X_train, X_test, Y_train, Y_test, list(X_train.columns),classifier)
        print(FS_lab_description)
        if len(FS_lab_description) != 1:

            classifier.fit(FS_new_train, Y_train.ravel())
            y_test_predict = classifier.predict(FS_new_test)

            accuracy = accuracy_score(Y_test, y_test_predict) * 100
            scores = cross_val_score(classifier, np.concatenate([FS_new_train, FS_new_test]), labels, cv=10)
            print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
            print("Accuracy: " + str(accuracy) + '%')

            all_accuracy += [accuracy]
            all_scores += [scores]
    #print(classification_report(Y_test, y_test_predict))

    print(np.mean(all_accuracy),np.std(all_accuracy))
    print(np.mean(all_scores),np.std(all_scores))

import matplotlib.pyplot as plt
def online(X_train, X_test, Y_train, Y_test):
    all_acc = []
    all_le = 0
    for fsx in range(len(X_train)):
        print(np.array(Y_train[fsx]).shape)

        print("=" * 30)
        # Predict Test Set
        favorite_clf = GradientBoostingClassifier()
        favorite_clf.fit(X_train[fsx], Y_train[fsx])
        level2 = []
        #test phase
        test_predictions = favorite_clf.predict(X_test[fsx])
        test_proba = favorite_clf.predict_proba(X_test[fsx])
        print(test_predictions)
        y_test_num = [1 if yt == 'Breathe' else 2 if yt == 'Apneia' else 0 for yt in Y_test[fsx]]
        pred_num = [1 if pt == 'Breathe' else 2 if pt == 'Apneia' else 0 for pt in test_predictions]
        print(y_test_num)
        acc = accuracy_score(y_test_num, pred_num)
        le = len(X_test[fsx])

        print(classification_report(Y_test[fsx], test_predictions))

        print("Accuracy: {:.4%}".format(acc))
        print('+' * 20)
        all_acc += [acc*le]
        all_le += le
        X_test_ = np.array(pickle.load(open('_X_test.pickle', 'rb')))

        plt.plot(test_proba)
        plt.plot(X_test[fsx][0])
        plt.show()
    print('final accuracy ---- ' + str(np.sum(all_acc)*100/all_le))


online(FS_X_train, FS_X_test, Y_train_, Y_test_)
#regular(FS_X_train, Y_train_)