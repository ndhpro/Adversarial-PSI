import os
import json
import multiprocessing
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split


def split_data():
    X, y = list(), list()
    root = "../psi_graph/"
    for fd in ["bashlite/", "mirai/", "others/", "benign/"]:
        for _, _, files in os.walk(root + fd):
            for fname in files:
                X.append(fname.replace(".txt", ""))
                if fd == "benign/":
                    y.append(0)
                else:
                    y.append(1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=2020, stratify=y
    )
    pd.DataFrame({"name": X_train, "label": y_train}).to_csv(
        "data/train_list.csv", index=None)
    pd.DataFrame({"name": X_test, "label": y_test}).to_csv(
        "data/test_list.csv", index=None)


def split_data_adv():
    X, y = list(), list()
    train = pd.read_csv("data/train_list.csv")
    test = pd.read_csv("data/test_list.csv")
    data = pd.concat([train, test], ignore_index=True)
    root = "adversarial/"
    for _, _, files in os.walk(root):
        for fname in files:
            name = fname.replace(".atk.json", "")
            X.append(name)
            label = data.loc[data["name"]==name, "label"].values[0]
            y.append(label)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=2020, stratify=y
    )
    pd.DataFrame({"name": X_train, "label": y_train}).to_csv(
        "data/adv_train_list.csv", index=None)
    pd.DataFrame({"name": X_test, "label": y_test}).to_csv(
        "data/adv_test_list.csv", index=None)


def get_psig():
    def run(folder):
        path = "../psi_graph/" + folder
        for _, _, files in os.walk(path):
            for fname in files:
                node = dict()
                G = {"edges": list()}
                fpath = path + fname
                try:
                    with open(fpath, "r") as f:
                        data = f.read().split("\n")
                except Exception as e:
                    print(fname, e)
                    continue
                for line in data[2:-1]:
                    e = line.split()
                    if len(e) == 2:
                        if e[0] not in node:
                            node[e[0]] = len(node) + 1
                        if e[1] not in node:
                            node[e[1]] = len(node) + 1
                        G["edges"].append([node[e[0]], node[e[1]]])
                with open("psig/" + fname.replace(".txt", ".json"), "w") as f:
                    json.dump(G, f)

    num_cores = multiprocessing.cpu_count()
    folders = ["bashlite/", "mirai/", "others/", "benign/"]
    results = Parallel(n_jobs=num_cores, verbose=100)(
        delayed(run)(fd) for fd in folders)


if __name__ == "__main__":
    # split_data()
    # get_psig()
    split_data_adv()
