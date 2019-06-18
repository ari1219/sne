#-*- coding:utf-8 -*-

from simple_sne import SimpleSNE
import pandas as pd
import networkx as nx
import tensorflow as tf

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
G = nx.Graph()
G.add_edges_from(list(map(lambda x:[x[0], x[1]], edges)))
n = G.number_of_nodes()
if not directed:
    edges += [[edge[1], edge[0], edge[2]] for edge in edges]

sne = SimpleSNE(n, edges, d=2)
with tf.Session() as sess:
    sne.variables_initialize(sess)
    for _ in range(1000):
        sne.train_one_epoch(sess)
    x = sne.get_embedding(sess)
