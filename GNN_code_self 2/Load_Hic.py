import numpy as np
import scipy.sparse
import pandas as pd
import scipy
from sklearn.manifold import TSNE
from scipy.io import loadmat
import scipy
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets import make_blobs
import matplotlib.colors as mcolors
import networkx as nx
import matplotlib.pyplot as plt

src_dir = "/Users/shaydeutsch/Dropbox/Graph_Networxx/Hic/"
src_name = src_dir + "CASTH1_inter_10k_KR_chr1.npz"
#src_name = src_dir + "CASTH1_inter_50k_KR_chr1.npz" 
sparse_matrix = scipy.sparse.load_npz(src_name)
arr = sparse_matrix.toarray()
N=arr.shape[0]
arr2 = arr[300:N,300:N]
arr3=arr2
np.fill_diagonal(arr3, 0)
super_threshold_indices = arr3 < 10
arr3[super_threshold_indices] = 0
G =  nx.from_numpy_matrix(arr3)
nodelist = list(G)
A = nx.to_scipy_sparse_matrix(G, nodelist=nodelist)
n, m = A.shape
diags = A.sum(axis=1)
D = scipy.sparse.spdiags(diags.flatten(), [0], m, n, format="csr")
res = np.where(diags !=0)[0]
arr5=arr3
arr5=arr3[res,:]
arr5=arr5[:,res]
G =  nx.from_numpy_matrix(arr5)