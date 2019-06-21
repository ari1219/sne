# -*- coding:utf-8 -*-

import tensorflow as tf
import math

class SDME:

    def __init__(self, n, edges, d=60):
        self.n = n
        self.edges = edges
        self.dim = d
        self.build_valiables()
        self.build_placeholder()
        self.build_lookup()
        self.build_model()

    def build_valiables(self):
        self.bound = 6/math.sqrt(self.dim)
        self.emb_s = tf.Variable(tf.random_uniform([self.n, self.dim], minval=-self.bound, maxval=self.bound), dtype=tf.float32)
        self.emb_t = tf.Variable(tf.random_uniform([self.n, self.dim], minval=-self.bound, maxval=self.bound), dtype=tf.float32)
        self.m1 = tf.Variable(tf.random_uniform([self.n, self.dim], minval=-self.bound, maxval=self.bound), dtype=tf.float32)
        self.b1 = tf.Variable(tf.zeros([self.dim]), dtype=tf.float32)
        self.m2 = tf.Variable(tf.random_uniform([self.n, self.dim], minval=-self.bound, maxval=self.bound), dtype=tf.float32)

    def build_placeholder(self):
        self.s = tf.placeholder(tf.int64, shape=[None])
        self.t = tf.placeholder(tf.int64, shape=[None])
        self.r = tf.placeholder(tf.float32, shape=[None])

    def build_lookup(self):
        self.s_vec = tf.nn.embedding_lookup(self.emb_s, )

    def build_model(self):
