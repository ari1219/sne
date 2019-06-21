# -*- coding:utf-8 -*-

import tensorflow as tf
import math
import random

class Model:

    def __init__(self, n, edges, directed=True, alpha=1, k=5, d=60, lam=0.00001, lr=0.1, batch_size=1000):
        self.num_of_nodes = n
        self.edges = edges
        self.directed = directed
        self.alpha = alpha
        self.num_of_negative_sampling = k
        self.embedding_dim = d
        self.lam = lam
        self.learning_rate = lr
        self.batch_size = batch_size
        self.build_embedding()
        self.build_placeholder()
        self.build_lookup()
        self.build_model()
        self.build_train()

    def build_placeholder(self):
        self.s = tf.placeholder(tf.int64, shape=[None])
        self.t = tf.placeholder(tf.int64, shape=[None])
        self.r = tf.placeholder(tf.int64, shape=[None])
        self.reshaped_r = tf.reshape(tf.cast(self.r, dtype=tf.float32), shape=[-1, 1])

    def build_lookup(self):
        self.s_vec = tf.nn.embedding_lookup(self.emb_s, self.s)
        if self.directed == True:
            self.t_vec = tf.nn.embedding_lookup(self.emb_t, self.t)
        else:
            self.t_vec = tf.nn.embedding_lookup(self.emb_s, self.t)

    def build_embedding(self):
        self.bound = 6 / math.sqrt(self.embedding_dim)
        self.emb_s = tf.Variable(tf.random_uniform([self.num_of_nodes, self.embedding_dim], minval=-self.bound, maxval=self.bound), dtype=tf.float32)
        self.emb_t = tf.Variable(tf.random_uniform([self.num_of_nodes, self.embedding_dim], minval=-self.bound, maxval=self.bound), dtype=tf.float32)

    def build_model(self):
        self.build_loss_skip()
        self.build_loss_rel()
        self.build_loss_reg()
        self.total_loss = tf.reduce_mean(self.loss_skip + self.alpha * self.loss_rel) + self.lam * self.loss_reg

    def build_loss_skip(self):
        self.w_neg = tf.Variable(tf.random_uniform([self.embedding_dim, self.embedding_dim], minval=-self.bound, maxval=self.bound), dtype=tf.float32)
        self.w_pos = tf.Variable(tf.random_uniform([self.embedding_dim, self.embedding_dim], minval=-self.bound, maxval=self.bound), dtype=tf.float32)
        self.ws = (1 + tf.cast(self.reshaped_r, tf.float32))/2*tf.matmul(self.s_vec, self.w_pos) + (1-tf.cast(self.reshaped_r, tf.float32))/2*tf.matmul(self.s_vec, self.w_neg)
        self.skip_bias = tf.Variable(tf.zeros([self.num_of_nodes]), dtype=tf.float32)
        self.loss_skip = tf.nn.sampled_softmax_loss(
                self.emb_t,
                self.skip_bias,
                tf.reshape(self.t, shape=[-1, 1]),
                self.ws,
                self.num_of_negative_sampling,
                self.num_of_nodes
            )

    def build_loss_rel(self):
        self.loss_rel = self.reshaped_r*tf.sigmoid(tf.reduce_sum(self.s_vec*self.t_vec, axis=1))

    def build_loss_reg(self):
        self.loss_reg = tf.nn.l2_loss(self.emb_s) + tf.nn.l2_loss(self.emb_t) + tf.nn.l2_loss(self.w_neg) + tf.nn.l2_loss(self.w_pos)

    def build_train(self):
        self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.total_loss)

    def variables_initialize(self, sess):
        sess.run(tf.global_variables_initializer())

    def train_one_epoch(self, sess, ret_loss=True):
        loss = 0
        for s, t, r in self.sample_iterater():
            feed_dict={self.s:s, self.t:t, self.r:r}
            _, l = sess.run([self.train, self.total_loss], feed_dict=feed_dict)
            loss += l/(-(-self.num_of_nodes//self.batch_size))
        if ret_loss:
            return loss

    def calc_wpos_wneg_norm(self, sess):
        return sess.run(tf.nn.l2_loss(self.w_pos-self.w_neg))

    def sample_iterater(self):
        random.shuffle(self.edges)
        for i in range(0, len(self.edges), self.batch_size):
            s = [edge[0] for edge in self.edges[i:i+self.batch_size]]
            t = [edge[1] for edge in self.edges[i:i+self.batch_size]]
            r = [edge[2] for edge in self.edges[i:i+self.batch_size]]
            yield s, t, r

    def get_embedding(self, sess):
        return sess.run(self.emb_s)

if __name__ == "__main__":
    x = Model(100, [1, 1, 1])
