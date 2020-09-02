import json
import pickle
import numpy as np
import networkx as nx
from copy import deepcopy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from graph2vec import feature_extractor


class Environment():
    def __init__(self, graph, label, gname):
        self.init_graph = graph
        self.graph = deepcopy(self.init_graph)
        self.init_state = dict()
        self.label = label
        self.gname = gname

        avail_action = dict()
        for node in graph.nodes:
            adj = graph.adj[node]
            if node.startswith('_'):
                continue
            avail_action[node] = graph.out_degree[node]

        avail_action = sorted(avail_action.items(),
                              key=lambda kv: kv[1], reverse=True)
        self.avail_action = [k for (k, v) in avail_action][:10]
        # print(self.avail_action)

        self.d2v = Doc2Vec.load("d2v.model")
        self.fs = pickle.load(open("result/model/fs.pickle", "rb"))
        self.scaler = pickle.load(open("result/model/scaler.pickle", "rb"))
        self.clf = pickle.load(open("result/model/svm.pickle", "rb"))

    def reset(self):
        self.graph = deepcopy(self.init_graph)
        return self.init_state

    def get_avail_action(self):
        return self.avail_action

    def step(self, action, state, step):
        reward = -1
        # dummy = action + '+' + str(state.get(action, 0) + 1)
        dummy = action + '+'
        if dummy not in self.graph.nodes:
            self.graph.add_node(dummy)
        self.graph.add_edge(action, dummy)
        node = dict()
        G = {"edges": list()}
        for e in self.graph.edges:
            u, v, _ = e
            if u not in node:
                node[u] = len(node) + 1
            if v not in node:
                node[v] = len(node) + 1

            G["edges"].append([node[u], node[v]])
        with open("temp/" + self.gname + ".atk.json", "w") as f:
            json.dump(G, f)

        doc = feature_extractor("temp/" + self.gname + ".atk.json", rounds=3)
        x = self.d2v.infer_vector(doc.words).reshape(1, -1)
        x = self.fs.transform(x)
        x = self.scaler.transform(x)
        y = self.clf.predict(x)[0]

        if step == 49:
            reward = -1000
        if y == 1 - self.label:
            reward = 1000
            done = True
        else:
            done = False

        next_state = dict(state)
        next_state[action] = next_state.get(action, 0) + 1
        return next_state, reward, done
