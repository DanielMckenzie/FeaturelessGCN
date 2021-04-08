#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:51:21 2021

@author: shaydeutsch
"""
import torch
import numpy as np
from pygsp import graphs, filters, plotting, utils
import scipy
from sklearn.manifold import TSNE
from scipy.io import loadmat
import scipy
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets import make_blobs
import matplotlib.colors as mcolors
import networkx as nx
    



def load_benchmark_cora_no_features(dataset="pubmed"):#citeseer, pubmed, cora
    """
    Load benchmark dataset: citeseer, pubmed, cora
    all outputs are tensors
    REFER: https://github.com/kimiyoung/planetoid
    """
    print('Loading {} dataset...'.format(dataset))

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/benchmark/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/benchmark/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)
    
    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended


# efficient version
    features_ = sp.vstack((allx, tx)).tolil()
    features_[test_idx_reorder, :] = features_[test_idx_range, :]   
    features = sp.csr_matrix(features_).toarray()
    features = normalize_row (features)
    features = torch.FloatTensor(features)
    features_A = features.numpy()
    [m, n] = features_A.shape
    rr = np.random.rand(m, n)
    features = rr 
    features = normalize_row (features)
    features = torch.FloatTensor(features)
    
    
   
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + np.eye(features.shape[0])
    adj = torch.FloatTensor(adj)
    
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_train = list(idx_train)
    idx_val = range(len(y), len(y)+500)
    idx_val = list(idx_val)
    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    features = torch.FloatTensor(features)
    
 
    l = np.where(~labels.any(axis=1))[0]
   
    if len(l)>0:
        for i in range(len(l)):
            labels[l[i]][1] = 1
    
    labels = torch.LongTensor(np.where(labels)[1])
    labels = torch.LongTensor(labels)
    print("finish load data")
    return adj, features, labels, idx_train, idx_val, idx_test










def load_benchmark2(dataset="AWA"):
    
    annots = loadmat('/Users/shaydeutsch/Dropbox/Files/Tensor_Voting_Modify_toolbox/Tensor_Voting_junctions/MFD_Denoising/denoise_MFD_close_deploy/Data_for_GCN/AWA_W.mat')
    adj = annots['W_new']
    annots2 = loadmat('/Users/shaydeutsch/Dropbox/Files/Tensor_Voting_Modify_toolbox/Tensor_Voting_junctions/MFD_Denoising/denoise_MFD_close_deploy/Data_for_GCN/feats_denoised.mat')
    feat =   annots2['feats_denoised_new'] 
    feat = feat.transpose()
    annots3 = loadmat('/Users/shaydeutsch/Dropbox/Files/Tensor_Voting_Modify_toolbox/Tensor_Voting_junctions/MFD_Denoising/denoise_MFD_close_deploy/Data_for_GCN/labels_AWA.mat')
    labels = annots3['s_new']
    ##Load train index 
    annots4 = loadmat('/Users/shaydeutsch/Dropbox/Files/Tensor_Voting_Modify_toolbox/Tensor_Voting_junctions/MFD_Denoising/denoise_MFD_close_deploy/Data_for_GCN/train_idx.mat')
    idx_train =  annots4['train_labels_idx']   
    ##Load test index 
    annots5  =  loadmat('/Users/shaydeutsch/Dropbox/Files/Tensor_Voting_Modify_toolbox/Tensor_Voting_junctions/MFD_Denoising/denoise_MFD_close_deploy/Data_for_GCN/test_idx.mat')
    idx_test =  annots5['test_labels_idx']  
    ##Normalize adjacency matrix  
    #d = np.sum(adj_temp, axis=0)
    #d = d**-0.5
    #D_inv = np.eye(adj.shape[0])
    #np.fill_diagonal(D_inv, d)
    #A_hat = D_inv * adj_temp
    #A_hat = A_hat * D_inv
    G =  nx.from_numpy_matrix(adj)
    b = G.edges()
    A1 = np.zeros((np.shape(adj)))                
    for (k,v) in b.items():
        A1[k] = 1   
    
    A2 = np.zeros((np.shape(adj)))
    A2 = np.transpose(A1)
    A_unweighted = np.add(A1, A2)
    np.fill_diagonal(A_unweighted, 0)
    adj = A_unweighted 
    adj_temp = adj
    adj_temp = adj_temp + np.eye(adj.shape[0]) 
    adj = adj_temp
    #adj_temp = adj
    #adj_temp = adj_temp + np.eye(adj.shape[0]) 
   # D = np.sum(adj, axis=0)
    #D_inv = D**-0.5
    #D_inv = np.diag(D_inv)
    #a1 = adj_temp * D_inv
    #A_hat = D_inv * a1
    #adj = A_hat 
    idx_val = idx_test
    labels = np.int64(labels)
    labels = labels.transpose()
    labels= labels-1
    #labels = np.argmax(labels, 1)
    labels = labels.ravel()
    #labels =  labels.tolist()
    idx_train = np.float32(idx_train)
    idx_train =  np.float32(idx_train) - 1
   # idx_train = np.argmax(idx_train, 1)
    idx_train = idx_train.ravel()
    idx_test = np.float32(idx_test) - 1
    idx_test = idx_test.ravel()
    #idx_test = np.argmax(idx_test, 1)
    #idx_test = idx_test.tolist()
    idx_val =  np.float32(idx_val) - 1
    #idx_val = np.argmax(idx_val, 1)
    idx_val = idx_val.ravel()
    adj = torch.FloatTensor(adj)
    mean = feat.mean(0)
    var = feat.std(0)
    feat = (feat - mean) / (var + 1e-6)
    feat = torch.FloatTensor(feat)
    labels_1dim = np.zeros(adj.shape[0])

    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
 
    return adj, feat, labels, idx_train, idx_val, idx_test 

