from src.get_data import load_benchmark
from src.normalization import get_adj_feats
from src.args import get_args
from src.models import get_model
from src.utils import accuracy
from src.utils import Sy_Embeddings
from Hic.utils import Compute_Betweeness_Centrality
from Hic.utils import Dim_Red
from Spectral_des import compute_norm_lap
from numpy import linalg as LA
from src.Load_Cora_no_features import load_benchmark_cora_no_features
from numpy.linalg import matrix_power
from control.matlab import *
import torch.optim as optim
import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import random
import control
from control import dlyap

# load dataset
dataset_name = 'cora'
#dataset_name = 'citeseer'
# dataset_name = input('input dataset name: cora/citeseer/pubmed/...')
adj, feats, labels, idx_train, idx_val, idx_test = load_benchmark(dataset_name)
####Modify features 
# features_A = feats.numpy()
# [m, n] = features_A.shape
A = adj.numpy()
np.fill_diagonal(A, 0)
[m, n] = A.shape
""" Defualt: compute centrality mebedding Betweeness (otherwise use random features)
"""
embeddings_graph_network = 1
if embeddings_graph_network==1:
   print("compute betweeness centrality")
   [z, BC] = Compute_Betweeness_Centrality(A)
   L_norm = compute_norm_lap(z)
   L = compute_norm_lap(A)
   I = np.identity(m)    
#Sy_Embeddings(z, L_norm, I)
   print("Compute Sylvester features")
   X = dlyap(z,L_norm,I)
   X = X.transpose()
   feats = torch.FloatTensor(X)
""" Here - Optional compute dimensionality reduction using SVD
""" 
# dimension_reduced = 800
# X = Dim_Red(X,dimension_reduced)
# feats = torch.FloatTensor(X)
"""Finish dimensionality reduction (Optional)
"""
"""finish computing features using graph centrality; 
"""

""" Otherwise use feats initialzied randomly
"""

if  embeddings_graph_network==0:
    feats = torch.FloatTensor(X)
#adj, feats, labels, idx_train, idx_val, idx_test = load_benchmark2()
model_name = 'GCN'
#model_name = 'SGC'
args = get_args(model_opt = model_name, dataset = dataset_name)
# processing

nb_class = (torch.max(labels) + 1).numpy()
Y_onehot =  torch.zeros(labels.shape[0], nb_class).scatter_(1, labels.unsqueeze(-1), 1)

nb_each_class_train = torch.sum(Y_onehot[idx_train], dim = 0)
nb_each_class_inv_train = torch.tensor(np.power(nb_each_class_train.numpy(), -1).flatten())
nb_each_class_inv_mat_train = torch.diag(nb_each_class_inv_train)

nb_each_class_val = torch.sum(Y_onehot[idx_val], dim = 0)
nb_each_class_inv_val = torch.tensor(np.power(nb_each_class_val.numpy(), -1).flatten())
nb_each_class_inv_mat_val = torch.diag(nb_each_class_inv_val)

nb_each_class_test = torch.sum(Y_onehot[idx_test], dim = 0)
nb_each_class_inv_test = torch.tensor(np.power(nb_each_class_test.numpy(), -1).flatten())
nb_each_class_inv_mat_test = torch.diag(nb_each_class_inv_test)

# get model
model = get_model(model_opt = model_name, nfeat = feats.size(1), \
                  nclass = labels.max().item()+1, nhid = args.hidden, \
                  dropout = args.dropout, cuda = args.cuda, \
                  dataset = dataset_name, degree = args.degree)
# optimizer
optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

# if args.cuda:
#     if model_name!='AGNN' and model_name!='GIN':
#         model.cuda()
#         feats = feats.cuda()
#         adj = adj.cuda()
#         labels = labels.cuda()
#         idx_train = idx_train.cuda()
#         idx_val = idx_val.cuda()
#         idx_test = idx_test.cuda()
    
    
# Print model's state_dict    
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor,"\t",model.state_dict()[param_tensor].size()) 
print("optimizer's state_dict:")

# Print optimizer's state_dict
for var_name in optimizer.state_dict():
    print(var_name,"\t",optimizer.state_dict()[var_name])
    
# Print parameters
for name, param in model.named_parameters():
    print(name, param)
    
delt = 0.05

# train, test


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, fp1, fp2 = model(feats, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output, fp1, fp2 = model(feats, adj)
    
    CE_loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    loss_val = CE_loss_val
    acc_val = accuracy(output[idx_val], labels[idx_val])
    
    CE_loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    loss_test = CE_loss_test
    acc_test = accuracy(output[idx_test], labels[idx_test])
    
    
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'loss_test: {:.4f}'.format(loss_test.item()),
          'acc_test: {:.4f}'.format(acc_test.item()),
          'time: {:.4f}s'.format(time.time() - t))
    print("-------------------------------------------------")

    return epoch+1, loss_train.item(), acc_train.item(), loss_val.item(), \
            acc_val.item(), loss_test.item(), acc_test.item(), time.time() - t, \
            output, fp1, fp2
            
training_log = []

# Train model
t_total = time.time()
temp_val_loss = 999999
temp_val_acc = 0
temp_test_loss = 0
temp_test_acc = 0

for epoch in range(args.epochs):


    epo, trainloss, trainacc, valloss, valacc, testloss, testacc, epotime, output, fp1, fp2 = train(epoch)
    training_log.append([epo, trainloss, trainacc, valloss, valacc, testloss, testacc, epotime])

    if valacc >= temp_val_acc:
        temp_val_loss = valloss
        temp_val_acc = valacc
        temp_test_loss = testloss
        temp_test_acc = testacc
        propagation_feats = fp1
        before_softmax = fp2
        after_softmax = output
        
        if model_name == 'AFGNN':
            temp_weight = torch.softmax(model.state_dict()['gc1.linear_weight'].data,dim=0)

#from scipy.io import savemat
#from scipy.io import savemat
#scipy.io.savemat('/Users/shaydeutsch/Dropbox/Files/Tensor_Voting_Modify_toolbox/Tensor_Voting_junctions/MFD_Denoising/denoise_MFD_close_deploy/Data_for_GCN/Cora.GCN.mat', mdict={'output': output})
#from scipy.io import savemat
#scipy.io.savemat('/Users/shaydeutsch/Dropbox/Files/Tensor_Voting_Modify_toolbox/Tensor_Voting_junctions/MFD_Denoising/denoise_MFD_close_deploy/Data_for_GCN/Cora.labels.mat', mdict={'labels': labels})
#from scipy.io import savemat
#scipy.io.savemat('/Users/shaydeutsch/Dropbox/Files/Tensor_Voting_Modify_toolbox/Tensor_Voting_junctions/MFD_Denoising/denoise_MFD_close_deploy/Data_for_GCN/Cora.adj.mat', mdict={'adj': adj})
### Test with Poisson Learning: Graph Based Semi-Supervised Learning
import graphlearning as gl
idx_test = idx_test.numpy()
labels =   labels.numpy()
idx_train = idx_train.numpy()
num_labels = max(labels)+1
np.fill_diagonal(A, 0)
l_pois = gl.graph_ssl(A,idx_train,labels[idx_train],algorithm='poisson')
acc = gl.accuracy(l_pois[idx_test],labels[idx_test],num_labels)
labels_per_class = 20 #5 labels per class
#Compute accuracy
acc = gl.accuracy(l_pois[idx_test],labels[idx_test],labels_per_class)   
print("In compariosn: Poisson MBO: Accuracy=%f"%acc)





