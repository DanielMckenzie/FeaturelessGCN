import torch.nn as nn
import torch.nn.functional as F
import torch
from src.layers import AFGNNlayer, GraphConvolution, AFGNNlayer_fixed_order, FD1layer, FD2layer, GraphIsomorphism
import sys

class GCN(nn.Module):
    '''
    two layer GCN
    one normalized adj
    '''
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    
    def forward(self, x, adj):# x and adj are the input for this GCN module
        u1, fp1 = self.gc1(x, adj)
        u1 = F.relu(u1)
        u1 = F.dropout(u1, self.dropout, training=self.training)
        u2, fp2 = self.gc2(u1, adj)
        res = F.log_softmax(u2, dim = 1)
        return res, fp1, u2   

class SGC(nn.Module):
    """
    SGC layer
    x is propagated feature, adj is None
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(SGC, self).__init__()
        self.W1 = nn.Linear(nfeat, nclass)

    def forward(self, x, adj):
        fp = x
        x = self.W1(x)
        res = F.log_softmax(x, dim=1)
        return res, fp, 0

class GFNN(nn.Module):
    """
    gfnn layer
    x is propagated feature
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GFNN, self).__init__()
        self.W1 = nn.Linear(nfeat, nhid)
        self.W2 = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        fp = x
        x = F.relu(self.W1(x))
        x = self.W2(x)
        res = F.log_softmax(x, dim=1)
        return res, fp, 0

class GIN(nn.Module):
    '''
    two layer GIN
    one raw adj
    '''
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GIN, self).__init__()

        self.gi1 = GraphIsomorphism(nfeat, nhid)
        self.gi2 = GraphIsomorphism(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        u1, fp1 = self.gi1(x, adj)
        h1 = F.relu(u1)
        h1 = F.dropout(h1, self.dropout, training=self.training)
        u2, fp2 = self.gi2(h1, adj)
        res = F.log_softmax(u2, dim = 1)
        return res, fp1, fp2    
    
class BatchNorm(nn.Module):
    "Construct a batchnorm module."
    def __init__(self, features, eps=1e-6):
        super(BatchNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(0, keepdim=True)
        std  = x.std(0, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



class AFGNN(nn.Module):
    '''
    accommodative gnn
    adj is list of normalized adjacency matrix
    '''
    def __init__(self, nfeat, nclass, degree, dropout = 0):
        super(AFGNN, self).__init__()
        self.gc1 = AFGNNlayer(nfeat, nclass, degree)
        self.mapping = nn.Parameter(torch.FloatTensor(self.gc1.out_features, nclass))
        self.dropout = dropout
        self.reset_params()

    def reset_params(self):
        torch.nn.init.xavier_normal_(self.mapping)
          
    def norm(self, feature):
        assert len(feature.shape) == 2
        mean = feature.mean(dim = 0,keepdim = True)
        var = feature.std(dim = 0,keepdim = True)
        return (feature - mean) / (var + 1e-6)
          
    def forward(self, x, adj):# x and adj are the input for this GCN module
        u1, fp1 = self.gc1(x, adj)
        u1 = F.relu(u1)
        u1 = F.dropout(u1, self.dropout, training=self.training)
        u1 = torch.mm(u1, self.mapping)
        res = F.log_softmax(u1, dim=1)
        return res, fp1, u1

    
    
class FD1(nn.Module):
    '''
    accommodative gnn
    adj is list of normalized adjacency matrix
    '''
    def __init__(self, nfeat, nclass, degree, dropout = 0):
        super(FD1, self).__init__()
        self.gc1 = FD1layer(nfeat, nclass, degree)
        self.mapping = nn.Parameter(torch.FloatTensor(self.gc1.out_features, nclass))
        self.dropout = dropout
        self.reset_params()

    def reset_params(self):
        torch.nn.init.xavier_normal_(self.mapping)
          
    def norm(self, feature):
        assert len(feature.shape) == 2
        mean = feature.mean(dim = 0,keepdim = True)
        var = feature.std(dim = 0,keepdim = True)
        return (feature - mean) / (var + 1e-6)
          
    def forward(self, x, adj):# x and adj are the input for this GCN module
        u1, fp1 = self.gc1(x, adj)
        u1 = F.relu(u1)
        u1 = F.dropout(u1, self.dropout, training=self.training)
        u1 = torch.mm(u1, self.mapping)
        res = F.log_softmax(u1, dim=1)
        return res, fp1, u1
    
class FD2(nn.Module):
    '''
    accommodative gnn
    adj is list of normalized adjacency matrix
    '''
    def __init__(self, nfeat, nclass, degree, dropout = 0):
        super(FD2, self).__init__()
        self.gc1 = FD2layer(nfeat, nclass, degree)
        self.mapping = nn.Parameter(torch.FloatTensor(self.gc1.out_features, nclass))
        self.dropout = dropout
        self.reset_params()

    def reset_params(self):
        torch.nn.init.xavier_normal_(self.mapping)
          
    def norm(self, feature):
        assert len(feature.shape) == 2
        mean = feature.mean(dim = 0,keepdim = True)
        var = feature.std(dim = 0,keepdim = True)
        return (feature - mean) / (var + 1e-6)
          
    def forward(self, x, adj):# x and adj are the input for this GCN module
        u1, fp1 = self.gc1(x, adj)
        u1 = F.relu(u1)
        u1 = F.dropout(u1, self.dropout, training=self.training)
        u1 = torch.mm(u1, self.mapping)
        res = F.log_softmax(u1, dim=1)
        return res, fp1, u1   
    
# class AFGNN_pr(nn.Module):
#     '''
#     accommodative gnn
#     adj is list of normalized adjacency matrix
#     '''
#     def __init__(self, nfeat, nclass, degree, dropout = 0):
#         super(AFGNN_pr, self).__init__()
#         self.gc1 = AFGNN_filter(nfeat, nclass, degree)
#         self.reset_params()
          
#     def norm(self, feature):
#         assert len(feature.shape) == 2
#         mean = feature.mean(dim = 0,keepdim = True)
#         var = feature.std(dim = 0,keepdim = True)
#         return (feature - mean) / (var + 1e-6)
          
#     def forward(self, x, adj):# x and adj are the input for this GCN module
#         fp = self.gc1(x, adj)
#         return fp
    
    
    

class AFGNN_fixed_order(nn.Module):
    '''
    accommodative gnn
    adj is list of normalized adjacency matrix
    '''
    def __init__(self, nfeat, nclass, degree, dropout = 0.5):
        super(AFGNN_fixed_order, self).__init__()
        self.gc1 = AFGNNlayer_fixed_order(nfeat, nclass, degree)
        self.mapping = nn.Parameter(torch.FloatTensor(self.gc1.out_features, nclass))
        self.reset_params()

    def reset_params(self):
        torch.nn.init.xavier_normal_(self.mapping)
          
    def norm(self, feature):
        assert len(feature.shape) == 2
        mean = feature.mean(dim = 0,keepdim = True)
        var = feature.std(dim = 0,keepdim = True)
        return (feature - mean) / (var + 1e-6)
          
    def forward(self, x, adj):# x and adj are the input for this GCN module
        u1, fp1 = self.gc1(x, adj)
        u1 = F.relu(u1)
        u1 = torch.mm(u1, self.mapping)
        res = F.log_softmax(u1, dim=1)
        return res, fp1, 0

    
class AFGNN_fix_filter(nn.Module):
    """
    gfnn layer
    x is propagated feature
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.4):
        super(AFGNN_fix_filter, self).__init__()
        self.W1 = nn.Linear(nfeat, nhid)
        self.W2 = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        fp = x
        fpw = self.W1(x)
        x = F.relu(self.W1(x))
        x = self.W2(x)
        res = F.log_softmax(x, dim=1)
        return res, fp, fpw
    
    
    
def get_model(model_opt, nfeat, nclass, nhid, dropout, cuda, dataset, degree):
    if model_opt == "AFGNN":
        model = AFGNN(nfeat = nfeat,
                        nclass=nclass,
                        degree = degree)
    elif model_opt == "FD1":
        model = FD1(nfeat = nfeat,
                        nclass=nclass,
                        degree = degree)
    elif model_opt == "FD2":
        model = FD2(nfeat = nfeat,
                        nclass=nclass,
                        degree = degree)
    elif model_opt == "AFGNN_fix_filter":
        model = AFGNN_fix_filter(nfeat=nfeat,
                        nhid=nhid,
                        nclass=nclass)
    elif model_opt == "GFNN":
        model = GFNN(nfeat=nfeat,
                        nhid=nhid,
                        nclass=nclass,
                        dropout=dropout)
    elif model_opt == "GIN":
        model = GIN(nfeat=nfeat,
                        nhid=nhid,
                        nclass=nclass,
                        dropout=dropout)
    elif model_opt == 'AFGNN_fixed_order':
        model = AFGNN_fixed_order(nfeat = nfeat,
                        nclass=nclass,
                        degree = degree)
    elif model_opt == "GCN":
        model = GCN(nfeat=nfeat,
                    nhid=nhid,
                    nclass=nclass,
                    dropout=dropout)
    elif model_opt == "SGC":
        model = SGC(nfeat=nfeat,
                    nhid=nhid,
                    nclass=nclass,
                    dropout=dropout)
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))

    #if cuda: model.cuda()
    return model