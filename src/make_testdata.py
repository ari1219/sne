# -*- coding:utf-8 -*-

import networkx as nx
import random
import matplotlib.pyplot as plt
from simple_sne import SimpleSNE
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from relational_skipgram import Model
import numpy as np

G = nx.Graph()

n = 32
k = 2
pos_edge = 72
neg_edge = 144
for i in range(k):
    G.add_nodes_from(list(range(n*i, n*(i+1))))
    for _ in range(pos_edge):
        while True:
            edge = random.sample(list(range(n*i, n*(i+1))), 2)
            if not G.has_edge(edge[0], edge[1]):
                G.add_edge(edge[0], edge[1], sign=1, weight=0.9)
                break
for i in range(k-1):
    for j in range(i+1, k):
        for _ in range(neg_edge):
            while True:
                edge = [random.choice(list(range(n*i, n*(i+1)))), random.choice(list(range(n*j, n*(j+1))))]
                if not G.has_edge(edge[0], edge[1]):
                    G.add_edge(edge[0], edge[1], sign=-1, weight=0.1)
                    break
node_color = ["c"]*n + ["r"]*n# + ["b"]*n + ["y"]*n
edge_color = ["r" if G.edges[edge[0], edge[1]]["sign"]==1 else "b" for edge in G.edges]
node_size = 100

with open("data/sample.txt", "w") as f:
    for edge in G.edges:
        sign = G.edges[edge[0], edge[1]]["sign"]
        f.write(str(edge[0])+","+str(edge[1])+","+str(sign)+"\n")

# params
data = "data/sample.txt"
directed = True

# prepare data
with open(data, "r") as f:
    line = f.readline()
    edges = []
    while line:
        line = line.split("\n")[0]
        line = line.split(",")
        edges.append(line)
        line = f.readline()
n = G.number_of_nodes()
if not directed:
    edges += [[edge[1], edge[0], edge[2]] for edge in edges]

sne = Model(n, edges, d=10, directed=directed, alpha=0)
with tf.Session() as sess:
    sne.variables_initialize(sess)
    for i in range(40001):
        loss = sne.train_one_epoch(sess, ret_loss=True)
        if i % 20 == 0:
            w = sne.calc_wpos_wneg_norm(sess)
            print("epoch", i, ", loss=", loss, ", distance between w_pos and w_neg=", w)
        if i % 1000 == 0:
            x = sne.get_s_embedding(sess)
            y = sne.get_t_embedding(sess)
            x = np.concatenate([x, y], axis=1)
            pca = PCA(n_components=2)
            pca.fit(x)
            x = pca.transform(x)
            pos = dict()
            for node in G.nodes:
                pos[node] = x[node]
            plt.figure()
            nx.draw_networkx(G, pos=pos, node_size=node_size, node_color=node_color, width=0.5, edge_color=edge_color)
            plt.savefig("figure/figure_epoch"+str(i)+".png")
