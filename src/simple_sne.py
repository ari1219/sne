# -*- coding:utf-8 -*-

import tensorflow as tf
import random

class SimpleSNE:

    def __init__(self, n, edges, gamma=1, d=60, lr=0.0025, lam=0.001, batch_size=1000):
        self.n = n
        self.edges = edges
        self.gamma = gamma
        self.dim = d
        self.lr = lr
        self.lam = lam
        self.batch_size = 1000
        _bound = 6 / math.sqrt(self.embedding_dim)
        self.emb = tf.Variable(tf.random_uniform([n, d], minval=-_bound, maxval=_bound), dtype=tf.float32)
        self.s = tf.placeholder(tf.int32, shape=[None])
        self.t = tf.placeholder(tf.int32, shape=[None])
        self.r = tf.placeholder(tf.int32, shape=[None])
        self.neg_s = tf.placeholder(tf.int32, shape=[None])
        self.neg_t = tf.placeholder(tf.int32, shape=[None])
        self.s_emb = tf.nn.embedding_lookup(self.emb, self.s)
        self.t_emb = tf.nn.embedding_lookup(self.emb, self.s)
        self.neg_s_emb = tf.nn.embedding_lookup(self.emb, self.neg_s)
        self.neg_t_emb = tf.nn.embedding_lookup(self.emb, self.neg_t)
        self.sign = tf.cast(self.r, tf.float32)
        self.f_true = tf.norm(self.t_emb - self.r*self.s_emb, axis=1)
        self.f_r_neg = tf.norm(self.t_emb + self.r*self.s_emb, axis=1)
        self.f_t_neg = tf.norm(self.neg_t_emb - self.r*self.s_emb, axis=1)
        self.f_s_neg = tf.norm(self.t_emb - self.r*self.neg_s_emb, axis=1)
        self.loss_reg = tf.norm(self.emb)
        self.loss = tf.reduce_mean(tf.nn.relu(self.f_true+self.gamma-self.f_r_neg-self.f_t_neg-self.f_s_neg))+self.loss_reg
        self.train = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

    def variables_initialize(self, sess):
        sess.run(tf.global_valiables_initializer())

    def train_one_epoch(self, sess, ret_loss=True):
        loss = 0
        for s, t, r in sample_iterater():
            neg_s = np.random.randint(0, self.n, len(s))
            neg_t = np.random.randint(0, self.n, len(t))
            _, l = sess.run([self.train, self.loss], feed_dict={self.s:s, self.t:t, self.r:r, self.neg_s:neg_s, self.neg_t:neg_t})
            loss += l/(-(-self.n//self.batch_size))
        if ret_loss:
            return loss

    def sample_iterater(self):
        random.shuffle(self.edges)
        for i in range(0, len(self.edges), self.batch_size):
            s = [edge[0] for edge in edges[i:i+self.batch_size]]
            t = [edge[1] for edge in edges[i:i+self.batch_size]]
            r = [edge[2] for edge in edges[i:i+self.batch_size]]
            yield s, t, r
