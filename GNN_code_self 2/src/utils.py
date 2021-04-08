#from src.normalization import normalization
# from plots import plot_feature
import torch
# import cpnet
import networkx as nx
import math

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

#def mean_accuracy(output, labels):
#    preds = output.max(1)[1].type_as(labels)
#class_members == k
#    correct = preds.eq(labels).double()
#    for i in range(len(labels)):
#    temp= class_members(i)
#   correct = correct.sum()
#    return correct / len(labels)
def norm_expand_tensor(feature):
        assert len(feature.shape) == 2
        mean = feature.mean(dim=0,keepdim=True)
        var = feature.std(dim=0,keepdim=True)
        return (feature - mean) / (var + 1e-6)
    
def Sy_Embeddings(A, L, I):
     from scipy import linalg
     import networkx as nx
     x = linalg.solve_sylvester(A, L, I) 
     return x
    
def Core_Periphery(A):
    import networkx as nx
    import numpy as np
    G =  nx.from_numpy_matrix(A)
    #rb = cpnet.Rombach()
    rb = cpnet.LapCore()
    rb.detect(G)
    coreness = rb.get_coreness()
    coreness = list(coreness.items())
    coreness = np.array(coreness)
    return coreness

def Reg_Laplacian(A):
    import math
    import networkx as nx
    import numpy as np
    n, m = A.shape
    diags = A.sum(axis=1)
    diags = np.sqrt(diags)
    diag_inv =1/diags 
    diag=_sqrt[np.isinf(diag_sqrt)] = 0
    I = np.identity(m) 
    D = np.diagflat(diag_inv)
    A_norm = np.matmul(A, D)
    A_norm = np.matmul(D, A_norm)
    diags = A_norm.sum(axis=1)
    diags = np.sqrt(diags)
    diag_inv =1/diags 
    D_tilde = np.diagflat(diag_inv)
    L = D_tilde - A_norm 
    alpha  = 3
    Reg_L = np.linalg.pinv(I + alpha*L)
    #B = np.linalg.pinv(a)
    return L, Reg_L

