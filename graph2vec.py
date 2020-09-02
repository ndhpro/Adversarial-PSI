"""Graph2Vec module."""

import json
import glob
import hashlib
import pandas as pd
import networkx as nx
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """

    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])] + \
                sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + \
            list(new_features.values())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()


def dataset_reader(path):
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    name = path.replace(".json", "").split("/")[-1]
    try:
        data = json.load(open(path))
    except Exception as e:
        print(path, e)
        exit(0)
    graph = nx.from_edgelist(data["edges"])

    if "features" in data.keys():
        features = data["features"]
    else:
        features = nx.degree(graph)

    features = {int(k): v for k, v in features}
    return graph, features, name


def feature_extractor(path, rounds):
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    graph, features, name = dataset_reader(path)
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])
    return doc


def main():
    """
    Main function to read the graph list, extract features.
    Learn the embedding and save it.
    :param args: Object with the arguments.
    """
    # train = pd.read_csv("data/train_list.csv")
    # train["name"] = train["name"].apply(lambda x: "psig/" + x + ".json")
    # test = pd.read_csv("data/test_list.csv")
    # test["name"] = test["name"].apply(lambda x: "psig/" + x + ".json")

    train = pd.read_csv("data/adv_train_list.csv")
    train["name"] = train["name"].apply(lambda x: "adversarial/" + x + ".atk.json")
    test = pd.read_csv("data/adv_test_list.csv")
    test["name"] = test["name"].apply(lambda x: "adversarial/" + x + ".atk.json")

    print("Extracting features for training graphs...")
    graphs = train["name"].values
    num_cores = multiprocessing.cpu_count()
    document_collections = Parallel(n_jobs=num_cores)(
        delayed(feature_extractor)(g, rounds=3) for g in tqdm(graphs))

    print("Training doc2vec...")
    dimension = 64
    model = Doc2Vec(document_collections,
                    vector_size=dimension,
                    window=0,
                    dm=0,
                    negative=5,
                    workers=num_cores,
                    epochs=100,
                    alpha=0.025)
    model.save("d2v_adv.model")

    print("Save embeddings...")
    columns = ["name"]+["x_"+str(dim) for dim in range(dimension)] + ["label"]
    train_emb, test_emb = list(), list()
    for row in train.values:
        name = row[0].replace(".json", "").split("/")[-1]
        new_row = [name] + list(model.docvecs["g_"+name]) + [row[1]]
        train_emb.append(new_row)
    for row in test.values:
        name = row[0].replace(".json", "").split("/")[-1]
        new_row = [name] + list(model.infer_vector(
            feature_extractor(row[0], rounds=3).words)) + [row[1]]
        test_emb.append(new_row)

    train_emb = pd.DataFrame(train_emb, columns=columns).sort_values(["name"])
    train_emb.to_csv("data/train_embeddings_adv.csv", index=None)
    test_emb = pd.DataFrame(test_emb, columns=columns).sort_values(["name"])
    test_emb.to_csv("data/test_embeddings_adv.csv", index=None)
    print("Complete.")


if __name__ == "__main__":
    main()
