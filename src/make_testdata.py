# -*- coding:utf-8 -*-

import networkx as nx
import random
import matplotlib.pyplot as plt
from simple_sne import SimpleSNE
import pandas as pd
import tensorflow as tf
from sklearn.manifold import TSNE

G = nx.Graph()

n = 32
k = 4
pos_edge = 72
neg_edge = 48
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
node_color = ["c"]*n + ["r"]*n + ["b"]*n + ["y"]*n
edge_color = ["r" if G.edges[edge[0], edge[1]]["sign"]==1 else "b" for edge in G.edges]
node_size = 100

with open("data/sample.txt", "w") as f:
    for edge in G.edges:
        sign = G.edges[edge[0], edge[1]]["sign"]
        f.write(str(edge[0])+","+str(edge[1])+","+str(sign)+"\n")

# params
data = "data/sample.txt"
directed = False

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

sne = SimpleSNE(n, edges, d=20)
with tf.Session() as sess:
    sne.variables_initialize(sess)
    for i in range(2000):
        loss = sne.train_one_epoch(sess, ret_loss=True)
        if i % 100 == 0 and i > 1:
            x = sne.get_embedding(sess)
            x = TSNE(n_components=2).fit_transform(x)
            pos = dict()
            for node in G.nodes:
                pos[node] = x[node]
            plt.figure()
            nx.draw_networkx(G, pos=pos, node_size=node_size, node_color=node_color, width=0.5, edge_color=edge_color)
            plt.savefig("figure/figure_epoch"+str(i)+".png")
