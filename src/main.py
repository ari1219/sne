#-*- coding:utf-8 -*-

from simple_sne import SimpleSNE
import pandas as pd
import networkx as nx
import tensorflow as tf

# params
data = "data/Slashdot.csv"
directed = True

# prepare data
edges = pd.read_csv(data).values.tolist()
G.add_edge(list(map(lambda x:(x[0], x[1]),edges)))
n = G.number_of_nodes()
if not directed:
    edges += [[edge[1], edge[0], edge[2]] for edge in edges]

sne = SimpleSNE(n, edges)
with tf.Session() as sess:
    sne.variables_initialize(sess)
    sne.train_one_epoch(sess)
    x = sne.get_embedding(sess)
