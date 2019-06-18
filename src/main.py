#-*- coding:utf-8 -*-

from simple_sne import SimpleSNE
import pandas as pd
import networkx as nx
import tensorflow as tf

# params
data = "data/Slashdot.csv"

edges = pd.read_csv(data).values.tolist()
G = nx.Graph()
G.add_edges_from(list(map(lambda x:(x[0], x[1]),edges)))
n = G.number_of_nodes()

sne = SimpleSNE(n, edges)
with tf.Session() as sess:
    sne.variables_initialize(sess)
    sne.train_one_epoch(sess)
    x = sne.get_embedding(sess)
