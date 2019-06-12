# -*- coding:utf-8 -*-

import tensorflow as tf
import math

class Model:

    def __init__(self, n, alpha=10, k=10, d=60, lam=0.001, lr=0.0025):
        self.num_of_nodes = n
        self.alpha = alpha
        self.num_of_negative_sampling = k
        self.embedding_dim = d
        self.lam = lam
        self.learning_rate = lr
        bound = 6 / math.sqrt(self.embedding_dim)
        self.emb_s = tf.Variable(tf.random_uniform([num_of_nodes, d], minval=-bound, maxval=bound), dtype=tf.float32)
        self.emb_t = tf.Variable(tf.random_uniform([num_of_nodes, d], minval=-bound, maxval=bound), dtype=tf.float32)
        self.emb_r = tf.Variable(tf.ranodm_uniform([2, d], minval=-bound, maxval=bound), dtype=tf.float32)
        self.W_sr = tf.Variable(tf.random_uniform([d, d], minval=-bound, maxval=bound), dtype=tf.flaot32)
        self.W_st = tf.Variable(tf.ranodm_uniform([d, d], minval=bound, maxval=bound), dtype=tf.float32)

    def build_graph(self):
        self.loss_rel = self.build_loss_rel()
        self.loss_skip = self.build_loss_skip()
        self.loss_reg = self.build_loss_reg()
        self.total_loss = self.loss_skip + self.alpha * self.loss_rel + self.lam * self.loss_reg
        self.train = self.build_train()

    def build_loss_reg(self):
        self.
