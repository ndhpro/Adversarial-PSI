import pickle
import pandas as pd
from sklearn.metrics import accuracy_score


train = pd.read_csv("data/train_embeddings.csv")
test_ = pd.read_csv("data/test_embeddings.csv")
test = pd.concat([train, test_], ignore_index=True).sort_values(["name"])
x = test.loc[:, "x_0":"x_63"].values

fs = pickle.load(open("result_adv/model/fs.pickle", "rb"))
scaler = pickle.load(open("result_adv/model/scaler.pickle", "rb"))
clf = pickle.load(open("result_adv/model/svm.pickle", "rb"))
x = fs.transform(x)
x = scaler.transform(x)
y = clf.predict(x)
test_res = pd.DataFrame({
    "name": test["name"],
    "label": test["label"],
    "pred": y
}).to_csv("result_adv/test_res_adv.csv", index=None)
print(accuracy_score(test["label"].values, y))