import os
import sys
import time
import networkx as nx
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from env import Environment
from agent import QLearningTable


def load_graph(path):
    G = nx.MultiDiGraph()
    with open(path, 'r') as f:
        lines = f.readlines()

    for line in lines[2:]:
        line = line.strip()
        if ' ' in line and line.find(' ') == line.rfind(' '):
            (u, v) = line.split(' ')
            G.add_edge(u, v)

    return G


def update(num_episode, env, RL):
    avail_action = env.get_avail_action()
    if not avail_action:
        return False

    for e in range(num_episode):
        step = 0
        state = env.reset()

        while step < 20:
            action = RL.choose_action(str(state), avail_action)

            new_state, reward, done = env.step(action, state, step)

            RL.learn(str(state), action, reward, str(new_state))
            state = dict(new_state)

            if done:
                break
            step += 1
        if done:
            print(state)
            break
        # print(e, state, done)
    return done


def run_file(path, label):
    gname = path.split('/')[-1].replace(".txt", '')
    print(gname)
    G = load_graph(path)
    if len(G.edges) == 0:
        print('Graph has no edge')
        return

    G = load_graph(path)
    env = Environment(graph=G, label=label, gname=gname)
    RL = QLearningTable(actions=list(G.nodes))

    n_eps = 20
    done = update(n_eps, env, RL)
    if not done:
        print("Attack failed.")
        try:
            os.remove("temp/" + gname + ".atk.json")
        except:
            pass


if __name__ == "__main__":
    run_file(sys.argv[1], 1)
