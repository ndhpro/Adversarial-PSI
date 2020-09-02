import os
import sys
import subprocess
import time
import pandas as pd
from main import run_file


if __name__ == "__main__":
    test = pd.read_csv("result/test_res.csv")

    for line in test["name"].values[:2500]:
        fl = False
        label = test.loc[test["name"]==line, "label"].values[0]
        pred = test.loc[test["name"]==line, "pred"].values[0]
        if label != pred:
            continue

        for _, _, files in os.walk('temp/'):
            for file_ in files:
                if line in file_:
                    print('Complete')
                    fl = True

        if not fl:
            root = "/home/ais/Downloads/psi_graph/"
            for fd in ["bashlite/", "mirai/", "others/", "benign/"]:
                fpath = root + fd + line + ".txt"
                c = False
                if os.path.exists(fpath):
                    run_file(fpath, label)
                    c = True
                    break
            if not c:
                print("PSI graph does not exist.")
