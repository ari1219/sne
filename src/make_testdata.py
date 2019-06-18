# -*- coding:utf-8 -*-

import networkx as nx
import random
import matplotlib.pyplot as plt

G = nx.Graph()

n = 32
k = 4
pos_edge = 144
neg_edge = 96
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
print([edge[0] for edge in G.edges])
edge_color = ["r" if G.edges[edge[0], edge[1]]["sign"]==1 else "b" for edge in G.edges]
node_size = 100
nx.draw_networkx(G, node_size=node_size, node_color=node_color, width=0.5, edge_color=edge_color)
plt.show()
