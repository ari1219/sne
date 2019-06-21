# -*- coding:utf-8 -*-

import tensorflow as tf
import math

class Model:

    def __init__(self, n, directed=True, alpha=10, k=10, d=60, lam=0.001, lr=0.025):
        self.num_of_nodes = n
        self.directed = directed
        self.alpha = alpha
        self.num_of_negative_sampling = k
        self.embedding_dim = d
        self.lam = lam
        self.learning_rate = lr
        self.build_placeholder()
        self.build_valiable()
        self.build_model()
        self.build_train()
        self.build_output_embedding()

    def build_placeholder(self):
        self.s = tf.placeholder(tf.int32, shape=[None])
        self.t = tf.placeholder(tf.int32, shape=[None])
        self.r = tf.placeholder(tf.int32, shape=[None])


    def build_valiable(self):
        _bound = 6 / math.sqrt(self.embedding_dim)
        self.emb_s = tf.Variable(tf.random_uniform([num_of_nodes, d], minval=-_bound, maxval=_bound), dtype=tf.float32)
        self.emb_t = tf.Variable(tf.random_uniform([num_of_nodes, d], minval=-_bound, maxval=_bound), dtype=tf.float32)
        self.emb_r = tf.Variable(tf.ranodm_uniform([2, d], minval=-_bound, maxval=_bound), dtype=tf.float32)

    def build_model(self):
        self.build_loss_skip()
        self.build_loss_rel()
        self.build_loss_reg()
        self.total_loss = self.loss_skip + self.alpha * self.loss_rel + self.lam * self.loss_reg

    def build_loss_skip(self):
