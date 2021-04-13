#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 17:10:38 2021

@author: danielmckenzie

Testing Poisson learning
"""

import graphlearning as gl
from src.get_data import load_benchmark

# load data
dataset_name = 'cora'
adj, feats, labels, idx_train, idx_val, idx_test = load_benchmark(dataset_name)
labels =   labels.numpy()
num_labels = max(labels)+1
num_train_per_class = 5
idx_test = idx_test.numpy()
idx_train = idx_train.numpy()

labels_train = labels[idx_train]
labels_test = labels[idx_test]



# Test Poisson Learning
labels_laplace = gl.graph_ssl(adj, idx_train, labels_train, algorithm='laplace')
labels_poisson = gl.graph_ssl(adj, idx_train, labels_train, algorithm='poisson')

#Compute and print accuracy
print('Laplace learning: %.2f%%'%gl.accuracy(labels,labels_laplace,len(idx_train)))
print('Poisson learning: %.2f%%'%gl.accuracy(labels,labels_poisson,len(idx_train)))

