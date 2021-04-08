import numpy as np
import random
import sys
from sklearn import datasets
import pickle as pkl
import scipy.sparse as sp
import torch
import networkx as nx
from sklearn import preprocessing

def normalize_row(mx):#orginal
    """Row-normalize sparse matrix"""
#     mx here corresponds to adj matrix
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.#turn inf into 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index



def load_benchmark(dataset="pubmed"):#citeseer, pubmed, cora
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
 #######
    features_A = features.numpy()
    [m, n] = features_A.shape
    rr = np.random.rand(m, n)
    features = rr 
    features = normalize_row (features)
    features = torch.FloatTensor(features)   
    print("new")
    
   
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




def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data_air_traffic(dataset_str):
    """Read the data and preprocess the task information."""
    dataset_G = "./data/new_benchmark/{}-airports.edgelist".format(dataset_str)
    dataset_L = "./data/new_benchmark/labels-{}-airports.txt".format(dataset_str)
    label_raw, nodes = [], []
    with open(dataset_L, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            node, label = lines.split()
            if label == 'label': continue
            label_raw.append(int(label))
            nodes.append(int(node))
    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(label_raw)
    G = nx.read_edgelist(open(dataset_G, 'rb'), nodetype=int)
    adj = nx.adjacency_matrix(G, nodelist=nodes)
#     print(adj)
    features = sp.csr_matrix(adj).toarray()
#     features = normalize_row(features)
    labels_1dim = np.zeros(adj.shape[0])
    for i in range(adj.shape[0]):
        for j in range(4):
            if labels[i][j] == 1:
                labels_1dim[i] = j

    # Randomly split the train/validation/test set
    indices = np.arange(adj.shape[0]).astype('int32')
#     np.random.seed(10)
    np.random.shuffle(indices)
    idx_train = indices[:adj.shape[0] // 3]
#     print(idx_train)
    idx_val = indices[adj.shape[0] // 3: (2 * adj.shape[0]) // 3]
    idx_test = indices[(2 * adj.shape[0]) // 3:]
    
#     print(idx_train)
    

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # task information
    degreeNode = np.sum(adj, axis=1).A1
    degreeNode = degreeNode.astype(np.int32)
    degreeValues = set(degreeNode)

    neighbor_list = []
    degreeTasks = []
    adj = adj.todense()
    for value in degreeValues:
        degreePosition = [int(i) for i, v in enumerate(degreeNode) if v == value]
        degreeTasks.append((value, degreePosition))

        d_list = []
        for idx in degreePosition:
            neighs = [int(i) for i in range(adj.shape[0]) if adj[idx, i] > 0]
            d_list += neighs
        neighbor_list.append(d_list)
        assert len(d_list) == value * len(degreePosition), 'The neighbor lists are wrong!'
    
    
    adj = torch.FloatTensor(adj)
    features = torch.FloatTensor(features)
    labels_1dim = torch.LongTensor(labels_1dim)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    
#     return adj, features, labels_1dim, y_train, y_val, y_test, train_mask, val_mask, test_mask, degreeTasks, neighbor_list
    return adj,  features, labels_1dim, idx_train, idx_val, idx_test

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx
