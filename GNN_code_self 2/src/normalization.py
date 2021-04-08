import scipy.sparse as sp
import torch
import torch.nn.functional as F
import numpy as np

def normalize_row(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.#turn inf into 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_sym(adj):
    """Symmetrically normalize adjacency matrix."""
#     adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    mx = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
#     mx = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return mx

def normalize_col(mx):
    """Column-normalize sparse matrix"""
#     mx here corresponds to adj matrix
    colsum = np.array(mx.sum(-1))
    c_inv = np.power(colsum, -1).flatten()
    c_inv[np.isinf(c_inv)] = 0.
    c_mat_inv = sp.diags(c_inv)
    mx = mx.dot(c_mat_inv)
    return mx

def normalize_morerow(mx):
    """normalize adjacency matrix, D-0.75AD-0.25."""
#     adj = sp.coo_matrix(adj)
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.75).flatten()
    r_inv[np.isinf(r_inv)] = 0.#turn inf into 0
    r_mat_inv = sp.diags(r_inv)

    colsum = np.array(mx.sum(-1))
    c_inv = np.power(colsum, -0.25).flatten()
    c_inv[np.isinf(c_inv)] = 0.
    c_mat_inv = sp.diags(c_inv)

    mx = mx.dot(r_mat_inv).transpose().dot(c_mat_inv)
#     mx = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return mx

def normalize_morecol(mx):
    """normalize adjacency matrix, D-0.75AD-0.25."""
#     adj = sp.coo_matrix(adj)
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.25).flatten()
    r_inv[np.isinf(r_inv)] = 0.#turn inf into 0
    r_mat_inv = sp.diags(r_inv)

    colsum = np.array(mx.sum(-1))
    c_inv = np.power(colsum, -0.75).flatten()
    c_inv[np.isinf(c_inv)] = 0.
    c_mat_inv = sp.diags(c_inv)

    mx = mx.dot(r_mat_inv).transpose().dot(c_mat_inv)
#     mx = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return mx

def normalization(A, norm_method, num_power):
    '''
    input dense matrix A
    norm_method: row, col, sym
    num_power: do feature propagation how many times
    '''
    
    A = sp.coo_matrix(A)
    
    if norm_method == 'row':
        # print("row normalization, propagate ", num_power, " times.")
        A = normalize_row(A)
    elif norm_method == 'col':
        # print("column normalization, propagate ", num_power, " times.")
        A = normalize_col(A)
    elif norm_method == 'morerow':
        A = normalize_morerow(A)
    elif norm_method == 'morecol':
        A = normalize_morecol(A)
    else:
        # print("symmetric normalization, propagate ", num_power, " times.")
        A = normalize_sym(A)
    
    A = sp.csr_matrix(A)
    res = A
#     print(res)
    while (num_power>1):
#         print(num_power)
        res = res * A
        num_power = num_power - 1
#         print(res)
    return res

def get_list_of_normalized_adjs(A,degree):
    '''
    for AFGNN
    '''
    norm_method = ['row', 'col', 'sym']
    adjs = []
    for method in norm_method:
        for i in range(degree):
#             print(method,i+1)
            adj_temp = normalization(A, method, i+1)
            adj_temp = sparse_mx_to_torch_sparse_tensor(adj_temp)
            adjs.append(adj_temp)
    return adjs

def get_list_of_normalized_adjs_FD1(A,degree):
    '''
    for Filter Design Approach 1
    now we just do hard code for degree = 2 case
    need to be revised into a more generalized version later
    '''
    adjs = []
    
    adj_row1 = sparse_mx_to_torch_sparse_tensor(normalization(A, 'row', 1))
    adjs.append(adj_row1)
    
    adj_col1 = sparse_mx_to_torch_sparse_tensor(normalization(A, 'col', 1))
    adjs.append(adj_col1)
    
    adj_row2 = sparse_mx_to_torch_sparse_tensor(normalization(A, 'row', 2))
    adjs.append(adj_row2)
    
    adj_col2 = sparse_mx_to_torch_sparse_tensor(normalization(A, 'col', 2))
    adjs.append(adj_col2)
    
    adj_row_col = sparse_mx_to_torch_sparse_tensor(normalization(A, 'row', 1)*normalization(A, 'col', 1))
    adjs.append(adj_row_col)
    
    adj_col_row = sparse_mx_to_torch_sparse_tensor(normalization(A, 'col', 1)*normalization(A, 'row', 1))
    adjs.append(adj_col_row)
    
    return adjs

# def get_list_of_normalized_adjs_FD2(A,degree):
#     '''
#     for Filter Design Approach 2
#     under this case, degree is useless, cuz we will do degree part after linear combination
#     we do not have I at this stage
#     '''
#     norm_method = ['row', 'col']
#     adjs = []
#     for method in norm_method:
#         adj_temp = normalization(A, method, 1)
#         adj_temp = sparse_mx_to_torch_sparse_tensor(adj_temp)
#         adjs.append(adj_temp)
#     return adjs

def get_list_of_normalized_adjs_with_fixed_order(A,degree):
    '''
    for AFGNN, order is fixed (not from 0 to degree, only row/col/sym with a fixed order)
    '''
    norm_method = ['row', 'col', 'sym']
    adjs = []
    for method in norm_method:
        adj_temp = normalization(A, method, degree)
        adj_temp = sparse_mx_to_torch_sparse_tensor(adj_temp)
        adjs.append(adj_temp)
    return adjs

def sgc_precompute(features, adj, degree):
    # here adj means normalized adj
    for i in range(degree):
        features = torch.spmm(adj, features)
    return features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_adj_feats(adj, feats, model_opt, degree, set_norm_method='sym'):
    '''
    input adjacency, feature tensor
    norm_method is sym by default, do not forget to change it
    output required adj and feats for model_opt
    '''
    A = adj.numpy()
    if model_opt == 'GCN':
        A = normalization(A = A, norm_method = 'sym', num_power = 1)
        adj = sparse_mx_to_torch_sparse_tensor(A)
        print("for GCN, return sym_norm(A) and raw feats")
        return adj, feats
    elif model_opt == 'SGC':
        identity = sparse_mx_to_torch_sparse_tensor(sp.eye(len(A)))
        # A = normalization(A = A, norm_method = 'sym', num_power = 1)
        A = normalization(A = A, norm_method = set_norm_method, num_power = 1)
        adj = sparse_mx_to_torch_sparse_tensor(A)
        feats = sgc_precompute(features = feats, adj = adj, degree = degree)
        print("for SGC, return identity matrix and propagated feats")
        return identity, feats
    elif model_opt == 'GFNN':
        identity = sparse_mx_to_torch_sparse_tensor(sp.eye(len(A)))
        A = normalization(A = A, norm_method = 'sym', num_power = 1)
        adj = sparse_mx_to_torch_sparse_tensor(A)
        feats = sgc_precompute(features = feats, adj = adj, degree = degree)
        print("for GFNN, return identity matrix and propagated feats")
        return identity, feats
    elif model_opt == 'GFN':
        d = np.array(A.sum(1))
        d = np.reshape(d,(-1,len(d))).T
        adj = sparse_mx_to_torch_sparse_tensor(sp.eye(len(A)))
        A = normalization(A = A, norm_method = 'sym', num_power = 1)
        feat_spar = sp.coo_matrix(feats.numpy())
        
        gfn_list = []
        gfn_list.append(d)
        gfn_list.append(feat_spar)
        adj_temp = feat_spar
        for i in range(degree):
            adj_temp = A * adj_temp
            gfn_list.append(adj_temp)
        
        feats = sp.hstack(gfn_list)
        feats = sparse_mx_to_torch_sparse_tensor(feats).to_dense()
        #feats = torch.FloatTensor(np.array(feats.todense())).float()
        print("for GFN, return identity matrix and the concatenation of propagated feats")
        return adj, feats
    elif model_opt == 'GIN':
        A = sp.coo_matrix(A)
        adj = sparse_mx_to_torch_sparse_tensor(A)
        print("for GIN, return raw adj matrix and raw feats matrix")
        return adj, feats
    elif model_opt == 'AFGNN' or model_opt == 'AFGNN_ift':
        adj = get_list_of_normalized_adjs(A,degree)
        return adj, feats
    elif model_opt == 'AFGNN_fixed_order':
        adj = get_list_of_normalized_adjs_with_fixed_order(A,degree)
        return adj, feats
    elif model_opt == 'FD1':
        adj = get_list_of_normalized_adjs_FD1(A, degree)
        return adj, feats
    elif model_opt == 'FD2':
        adj = get_list_of_normalized_adjs_FD1(A, degree)
        return adj, feats
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))