# -*- coding:utf-8 -*-

import tensorflow as tf
import random
import math
import numpy as np

class SimpleSNE:

    def __init__(self, n, edges, gamma=10, d=60, lr=0.1, lam=0.0001, batch_size=1000):
        self.n = n
        self.edges = edges
        self.gamma = gamma
        self.dim = d
        self.lr = lr
        self.lam = lam
        self.batch_size = 1000
        _bound = 6 / math.sqrt(self.dim)
        self.emb = tf.nn.l2_normalize(tf.Variable(tf.random_uniform([self.n, self.dim], minval=-_bound, maxval=_bound), dtype=tf.float32))
        self.s = tf.placeholder(tf.int32, shape=[None])
        self.t = tf.placeholder(tf.int32, shape=[None])
        self.r = tf.placeholder(tf.int32, shape=[None])
        self.neg_s = tf.placeholder(tf.int32, shape=[None])
        self.neg_t = tf.placeholder(tf.int32, shape=[None])
        self.s_emb = tf.nn.embedding_lookup(self.emb, self.s)
        self.t_emb = tf.nn.embedding_lookup(self.emb, self.t)
        self.neg_s_emb = tf.nn.embedding_lookup(self.emb, self.neg_s)
        self.neg_t_emb = tf.nn.embedding_lookup(self.emb, self.neg_t)
        self.sign = tf.reshape(tf.cast(self.r, tf.float32), shape=[-1, 1])
        self.f_true = tf.reduce_sum(tf.square(self.t_emb - self.s_emb*self.sign), axis=1)
        self.f_r_neg = tf.reduce_sum(tf.square(self.t_emb + self.s_emb*self.sign), axis=1)
        self.f_t_neg = tf.reduce_sum(tf.square(self.neg_t_emb - self.s_emb*self.sign), axis=1)
        self.f_s_neg = tf.reduce_sum(tf.square(self.t_emb - self.neg_s_emb*self.sign), axis=1)
        self.loss_reg = tf.nn.l2_loss(self.emb)
        self.loss = tf.reduce_mean(tf.nn.relu(3*self.f_true+self.gamma-self.f_r_neg-self.f_t_neg-self.f_s_neg))+self.lam*self.loss_reg
        self.train = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

    def variables_initialize(self, sess):
        sess.run(tf.global_variables_initializer())

    def train_one_epoch(self, sess, ret_loss=True):
        loss = 0
        for s, t, r in self.sample_iterater():
            neg_s = np.random.randint(0, self.n, len(s))
            neg_t = np.random.randint(0, self.n, len(t))
            feed_dict={self.s:s, self.t:t, self.r:r, self.neg_s:neg_s, self.neg_t:neg_t}
            _, l = sess.run([self.train, self.loss], feed_dict=feed_dict)
            loss += l/(-(-self.n//self.batch_size))
        if ret_loss:
            return loss

    def sample_iterater(self):
        random.shuffle(self.edges)
        for i in range(0, len(self.edges), self.batch_size):
            s = [edge[0] for edge in self.edges[i:i+self.batch_size]]
            t = [edge[1] for edge in self.edges[i:i+self.batch_size]]
            r = [edge[2] for edge in self.edges[i:i+self.batch_size]]
            yield s, t, r

    def get_embedding(self, sess):
        return sess.run(self.emb)
