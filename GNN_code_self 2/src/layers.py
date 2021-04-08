import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    """
    Simple GCN layer
    paper: https://arxiv.org/abs/1609.02907
    refer: https://github.com/tkipf/pygcn/tree/master/pygcn
    need: normalized adj(sparse tensor), features(dense tensor)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        representation_aft_fp = torch.spmm(adj, input)
        output = torch.mm(representation_aft_fp, self.weight)
        if self.bias is not None:
            return output + self.bias, representation_aft_fp
        else:
            return output, representation_aft_fp

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphIsomorphism(Module):
    """
    Simple GIN layer
    paper: https://openreview.net/pdf?id=ryGs6iA5Km
    need: unnormalized adj(sparse matrix), features(dense tensor)
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(GraphIsomorphism, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.epsilon = Parameter(torch.FloatTensor(1))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.epsilon.data.fill_(1) 
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        ei = self.epsilon * torch.eye(adj.shape[0])
        representation_aft_fp = torch.spmm(adj, input) + torch.mm(ei, input)
        output = torch.mm(representation_aft_fp, self.weight)
        if self.bias is not None:
            return output + self.bias, representation_aft_fp
        else:
            return output, representation_aft_fp

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class AFGNNlayer(Module):
    """
    AFGNN layer
    need: a list of normalized adj(sparse matrix), features(dense tensor)
    """
    def __init__(self, in_features, out_features,degree, bias=True):
        super(AFGNNlayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.linear_weight = Parameter(torch.FloatTensor(3*degree+1))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(degree)

    def reset_parameters(self,degree):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.linear_weight.data.fill_(1/(3*degree+1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
            
    def forward(self, input, adj):# adj is a list of 2D matrix
        
#         step: XW, do this first to speed up
        out = [input.unsqueeze(0)]
        
#         step: list of AX
        for i in range(len(adj)):
            out.append(torch.spmm(adj[i], input).unsqueeze(0))


#         step: flatten each AX
        n = out[-1].shape[1]
        d = out[-1].shape[2]
        output = torch.cat(out,dim=0).view(-1,n*d).t()

        
#         step: weighted sum of all AX
        representation_aft_fp = torch.mm(output, torch.softmax(self.linear_weight,dim=0).unsqueeze(-1)).squeeze(-1).view(n,d)

# #         step: AXW
        output = torch.mm(representation_aft_fp, self.weight)
#         print("linear_weight: ", torch.softmax(self.linear_weight,dim=0))
        
        if self.bias is not None:
            return output + self.bias, representation_aft_fp
        else:
            return output, representation_aft_fp

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# class AFGNN_filter(Module):
#     """
#     AFGNN layer
#     need: a list of normalized adj(sparse matrix), features(dense tensor)
#     """
#     def __init__(self, in_features, out_features,degree, bias=False):
#         super(AFGNN_filter, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.linear_weight = Parameter(torch.FloatTensor(3*degree+1))
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters(degree)

#     def reset_parameters(self,degree):
#         self.linear_weight.data.fill_(1/(3*degree+1))
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
        
            
#     def forward(self, input, adj):# adj is a list of 2D matrix
        
# #         step: XW, do this first to speed up
#         out = [input.unsqueeze(0)]
        
# #         step: list of AX
#         for i in range(len(adj)):
#             out.append(torch.spmm(adj[i], input).unsqueeze(0))

# #         step: flatten each AX
#         n = out[-1].shape[1]
#         d = out[-1].shape[2]
#         output = torch.cat(out,dim=0).view(-1,n*d).t()

# #         step: weighted sum of all AX
#         representation_aft_fp = torch.mm(output, torch.softmax(self.linear_weight,dim=0).unsqueeze(-1)).squeeze(-1).view(n,d)

#         return representation_aft_fp

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'
    

class AFGNNlayer_fixed_order(Module):
    """
    AFGNN layer
    need: a list of normalized adj(sparse matrix), features(dense tensor)
    """
    def __init__(self, in_features, out_features,degree, bias=True):
        super(AFGNNlayer_fixed_order, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.linear_weight = Parameter(torch.FloatTensor(4))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(degree)

    def reset_parameters(self,degree):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.linear_weight.data.fill_(1/4)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
            
    def forward(self, input, adj):# adj is a list of 2D matrix
        
#         step: XW, do this first to speed up
        out = [input.unsqueeze(0)]
        
#         step: list of AX
        for i in range(len(adj)):
            out.append(torch.spmm(adj[i], input).unsqueeze(0))


#         step: flatten each AX
        n = out[-1].shape[1]
        d = out[-1].shape[2]
        output = torch.cat(out,dim=0).view(-1,n*d).t()

        
#         step: weighted sum of all AX
        representation_aft_fp = torch.mm(output, torch.softmax(self.linear_weight,dim=0).unsqueeze(-1)).squeeze(-1).view(n,d)

# #         step: AXW
        output = torch.mm(representation_aft_fp, self.weight)
#         print("linear_weight: ", torch.softmax(self.linear_weight,dim=0))
        
        if self.bias is not None:
            return output + self.bias, representation_aft_fp
        else:
            return output, representation_aft_fp

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class FD1layer(Module):
    """
    FD1 layer
    hard code for degree=2 case
    need: a list of normalized adj(sparse matrix), features(dense tensor)
    """
    def __init__(self, in_features, out_features,degree, bias=True):
        super(FD1layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.linear_weight = Parameter(torch.FloatTensor(3*degree+1))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(degree)

    def reset_parameters(self,degree):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.linear_weight.data.fill_(1/(3*degree+1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
            
    def forward(self, input, adj):# adj is a list of 2D matrix
        
#         step: XW, do this first to speed up
        out = [input.unsqueeze(0)]
        
#         step: list of AX
        for i in range(len(adj)):
            out.append(torch.spmm(adj[i], input).unsqueeze(0))


#         step: flatten each AX
        n = out[-1].shape[1]
        d = out[-1].shape[2]
        output = torch.cat(out,dim=0).view(-1,n*d).t()

        
#         step: weighted sum of all AX
        representation_aft_fp = torch.mm(output, torch.softmax(self.linear_weight,dim=0).unsqueeze(-1)).squeeze(-1).view(n,d)

# #         step: AXW
        output = torch.mm(representation_aft_fp, self.weight)
#         print("linear_weight: ", torch.softmax(self.linear_weight,dim=0))
        
        if self.bias is not None:
            return output + self.bias, representation_aft_fp
        else:
            return output, representation_aft_fp

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class FD2layer(Module):
    """
    FD2 layer
    hard code for degree = 2 case
    need: a list of normalized adj(sparse matrix), features(dense tensor)
    """
    def __init__(self, in_features, out_features,degree, bias=True):
        super(FD2layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.linear_weight = Parameter(torch.FloatTensor(3))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(degree)

    def reset_parameters(self,degree):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.linear_weight.data.fill_(1/3)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
            
    def forward(self, input, adj):# adj is a list of 2D matrix
        
#         step: X, do this first to speed up
        out = [input.unsqueeze(0)]
        
#         step: list of AX
        for i in range(len(adj)):
            out.append(torch.spmm(adj[i], input).unsqueeze(0))


#         step: flatten each AX
        n = out[-1].shape[1]
        d = out[-1].shape[2]
        output = torch.cat(out,dim=0).view(-1,n*d).t()
        
        softmax_val =  torch.softmax(self.linear_weight,dim=0)
        corresponding_weights = torch.FloatTensor(7)
        corresponding_weights[0] = softmax_val[0]*softmax_val[0]
        corresponding_weights[1] = 2*softmax_val[1]*softmax_val[0]
        corresponding_weights[2] = 2*softmax_val[2]*softmax_val[0]
        corresponding_weights[3] = softmax_val[1]*softmax_val[1]
        corresponding_weights[4] = softmax_val[2]*softmax_val[2]
        corresponding_weights[5] = softmax_val[1]*softmax_val[2]
        corresponding_weights[6] = softmax_val[2]*softmax_val[1]
        
        print("softmax_val: ", softmax_val)
        print("corresponding_weights: ", corresponding_weights)
        

        
#         step: weighted sum of all AX
        representation_aft_fp = torch.mm(output, corresponding_weights.unsqueeze(-1)).squeeze(-1).view(n,d)

# #         step: AXW
        output = torch.mm(representation_aft_fp, self.weight)
#         print("linear_weight: ", torch.softmax(self.linear_weight,dim=0))
        
        if self.bias is not None:
            return output + self.bias, representation_aft_fp
        else:
            return output, representation_aft_fp

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

