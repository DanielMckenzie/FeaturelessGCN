from src.get_data import load_benchmark
from src.normalization import get_adj_feats
from src.args import get_args
from src.Load_Hic import load_hic
from src.models import get_model
from src.utils import accuracy
from src.utils import Core_Periphery
import torch.optim as optim
import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import random
import networkx as nx
import scipy
from scipy.sparse import spdiags
from src.utils import Sy_Embeddings
import scipy
from scipy.io import savemat

dataset_name = 'Hic'
model_name = 'GCN'
args = get_args(model_opt = model_name, dataset = dataset_name)

A, edges = load_hic()
ab = np.argwhere(np.isnan(A))
A1=A
for i in range(ab.shape[0]):
    k1= ab[i, 0]
    k2= ab[i, 1]
    print(k1)
    A1[k1,k2] = 0
    
#coreness = Core_Periphery(A)
#idx = max(edges[1:500,1])
#idx = int(idx)
#A1= A1[0:idx,0:idx]
idx1= np.arange(201,303)
np.transpose(idx1)
A1= A[idx1,:]
A1= A1[:,idx1]
loop_idx_edges= np.arange(207,214)
np.transpose(loop_idx_edges)
edge_new = edges[loop_idx_edges,:]
edge_new = edge_new - 1 
A1 = A1[~np.all(A1 == 0, axis=1)]
#A1 = A1[~np.all(A1 == 0, axis=0)]
n, m = A1.shape
diags = A1.sum(axis=1) 
diags = np.sqrt(diags)
diag_inv =1/diags 
D = np.diagflat(diag_inv) 
L = D - A1 
I = np.identity(m) 
diags_sqrt = 1.0 / np.sqrt(diags)
diags_sqrt[np.isinf(diags_sqrt)] = 0

coreness = Core_Periphery(A1)
scipy.io.savemat('/Users/shaydeutsch/Dropbox/COIVD_19_proteins/HiC_netwrok/sub_A.mat', mdict={'A1': A1})
scipy.io.savemat('/Users/shaydeutsch/Dropbox/COIVD_19_proteins/HiC_netwrok/coreness.mat', mdict={'coreness': coreness})
#x = linalg.solve_sylvester(a, b, q)
scipy.io.savemat('/Users/shaydeutsch/Dropbox/COIVD_19_proteins/HiC_netwrok/edge_new.mat', mdict={'edge_new': edge_new})


