import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
# from imblearn.over_sampling import SMOTE


def report(names, y_true, y_pred, train_t, test_t):
    columns = ["Classifier", "Accuracy", "ROC AUC", "FPR", "Precision", "Recall",
               "F1-score", "Training time", "Testing time", "TP", "FP", "TN", "FN"]
    report = list()
    for name in names:
        cnf_matrix = metrics.confusion_matrix(y_true[name], y_pred[name])
        TN, FP, FN, TP = cnf_matrix.ravel()
        FPR = FP / (FP + TN)

        row = [
            str(name),
            round(100 * metrics.accuracy_score(y_true[name], y_pred[name]), 2),
            round(100 * metrics.roc_auc_score(y_true[name], y_pred[name]), 2),
            round(100 * FPR, 2),
            round(
                100 * metrics.precision_score(y_true[name], y_pred[name]), 2),
            round(100 * metrics.recall_score(y_true[name], y_pred[name]), 2),
            round(100 * metrics.f1_score(y_true[name], y_pred[name]), 2),
            round(train_t[name], 2),
            round(test_t[name], 2),
        ]
        row.extend([TP, FP, TN, FN])
        report.append(row)

    pd.DataFrame(report, columns=columns).to_csv(
        "result_adv/result.csv", index=None)


def draw_roc(names, colors, y_true, y_pred):
    plt.figure()
    for name, color in zip(names, colors):
        fpr, tpr, _ = metrics.roc_curve(y_true[name], y_pred[name])
        auc = metrics.roc_auc_score(y_true[name], y_pred[name])
        plt.plot(fpr, tpr, color=color,
                 label="%s (AUC = %0.2f)" % (name, auc))
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1-Specificity(False Positive Rate)")
    plt.ylabel("Sensitivity(True Positive Rate)")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(f"result_adv/roc.png", dpi=300)


def preprocessing():
    train = pd.read_csv("data/train_embeddings.csv")
    train_adv = pd.read_csv("data/train_embeddings_adv.csv")
    train = pd.concat([train, train_adv], ignore_index=True)
    test = pd.read_csv("data/test_embeddings.csv")
    test_adv = pd.read_csv("data/test_embeddings_adv.csv")
    test = pd.concat([test, test_adv], ignore_index=True)

    X_train, y_train = train.values[:, 1:-1], train.values[:, -1].astype("int")
    X_test, y_test = test.values[:, 1:-1], test.values[:, -1].astype("int")

    print("Data shape:", X_train.shape, X_test.shape)
    # sm = SMOTE(random_state=2020)
    # X_train, y_train = sm.fit_resample(X_train, y_train)

    fs = SelectFromModel(
        LinearSVC(penalty="l1", dual=False, random_state=2020).fit(X_train, y_train), prefit=True)
    X_train = fs.transform(X_train)
    X_test = fs.transform(X_test)
    pickle.dump(fs, open(f'result_adv/model/fs.pickle', 'wb'))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    pickle.dump(scaler, open(f'result_adv/model/scaler.pickle', 'wb'))
    return X_train, X_test, y_train, y_test


def classify(X_train, X_test, y_train, y_test):
    print("Processed data shape: ", X_train.shape, X_test.shape)
    y_true, y_pred, train_t, test_t = {}, {}, {}, {}

    names = ["Naive Bayes", "Decision Tree",
             "k-Nearest Neighbors", "SVM", "Random Forest"]
    fnames = ["nb", "dt", "knn",  "svm", "rf"]

    classifiers = [
        GaussianNB(),
        DecisionTreeClassifier(random_state=2020,
                               class_weight="balanced"),
        KNeighborsClassifier(n_jobs=-1),
        SVC(random_state=2020, class_weight="balanced"),
        RandomForestClassifier(random_state=2020,
                               class_weight="balanced", n_jobs=-1),
    ]

    hyperparam = [
        {},
        {"criterion": ["gini", "entropy"]},
        {"n_neighbors": [5, 100, 500], "weights": ["uniform", "distance"]},
        {"C": np.logspace(-3, 3, 7), "gamma": np.logspace(-3, 3, 7)},
        {"criterion": ["gini", "entropy"], "n_estimators": [10, 100, 1000]},
    ]

    colors = ["blue", "orange", "green", "red",
              "purple", "brown", "pink", "gray"]

    for name, fname, est, hyper in zip(names, fnames, classifiers, hyperparam):
        print(f"Running {name}...", end=' ', flush=True)
        clf = GridSearchCV(est, hyper, cv=5, n_jobs=-1)

        t = time()
        clf.fit(X_train, y_train)
        train_t[name] = time()-t

        t = time()
        y_true[name],  y_pred[name] = y_test, clf.predict(X_test)
        test_t[name] = time()-t

        acc = 100 * metrics.accuracy_score(y_true[name], y_pred[name])
        print("Accuracy: %0.2f." % acc)

        pickle.dump(clf, open(f"result_adv/model/{fname}.pickle", "wb"))

    report(names, y_true, y_pred, train_t, test_t)
    draw_roc(names, colors, y_true, y_pred)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocessing()
    classify(X_train, X_test, y_train, y_test)
