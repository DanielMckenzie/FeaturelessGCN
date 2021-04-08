#from src.normalization import normalization
# from plots import plot_feature
import torch
# import cpnet
import networkx as nx
import math
import numpy as np

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
    import math
    
    G =  nx.from_numpy_matrix(A)
    rb = cpnet.Rombach()
    #rb = cpnet.LapCore()
    rb = cpnet.LapSgnCore
    rb.detect(G)
    coreness = rb.get_coreness()
    coreness = list(coreness.items())
    coreness = np.array(coreness)
    return coreness

def Reg_Laplacian(A):
    import networkx as nx
    import numpy as np
    import math
    
    n, m = A.shape
    diags = A.sum(axis=1)
    diags = np.sqrt(diags)
    diag_inv =1/diags 
    #diag= np.sqrt[np.isinf(diag_inv)] = 0
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


def hypergeom_pmf(N, A, n, x):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpmath import mp
    from scipy.special import comb
  

    Achoosex = comb(A,x)
    NAchoosenx = comb(N-A, n-x)
    Nchoosen = comb(N,n)
    output =(Achoosex)*NAchoosenx/Nchoosen
    #prob12 = np.logaddexp(prob1, prob2)
    return output

def hypergeom_cdf(N, A, n, t):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import comb
    from mpmath import mp
    '''
    Cumulative Density Funtion for Hypergeometric Distribution
    :param N: population size
    :param A: total number of desired items in N
    :param n: number of draws made from N
    :param t: number of desired items in our draw of n items up to t
    :returns: CDF computed up to t
    '''
   # if min_value:
  #      return np.sum([hypergeom_pmf(N, A, n, x) for x in range(min_value, t+1)])
    
    return np.sum([hypergeom_pmf(N, A, n, x) for x in range(t+1)])

def hypergeom_plot(N, A, n, i):
    
    '''
    Visualization of Hypergeometric Distribution for given parameters
    :param N: population size
    :param A: total number of desired items in N
    :param n: number of draws made from N
    :returns: Plot of Hypergeometric Distribution for given parameters
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    
    x = np.arange(0, n+1)
    y = [hypergeom_pmf(N, A, n, x) for x in range(n+1)]
    plt.plot(x, y, 'bo')
    plt.plot(i,0.0001,'r+')
    plt.vlines(x, 0, y, lw=2)
    plt.vlines(i, 0, 0.0001, lw=10000),
    # plt.annotate('number of successes', xy=(i, 0.01), xytext=(0.1, 0.5),
    #        arrowprops=dict(facecolor='black'),
    #       )
    plt.annotate('number of successes', xy=(i, 0.0001), xytext=(0.01, 0.05),
            arrowprops=dict(facecolor='black'),
          )
    plt.xlabel('# of desired items in our draw')
    plt.ylabel('Probablities')
    plt.title('Hypergeometric Distribution Plot')
    plt.show()
    
    
    
def Diffusion_Be_Linear(Diffusion_score, X):
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    import random
    
    dim =  X.shape[1]
    num_points = X.shape[0]
    scores_per_removed_dim = np.zeros(dim)
    y = Diffusion_score
    y1 = np.arange(num_points)
    np.random.shuffle(y1)
    y2 = Diffusion_score[y1]
    for i in range(0,dim):
        if i==0: 
           X1 = X[:,i:]
        elif i==dim-1:
           X1 = X[:,0:dim-1]
        elif i>0 and i<dim-1:
           X11 = X[:,:i]
           X22 = X[:,i+1:]
           X1 = np.concatenate((X11, X22), axis=1)
           reg = LinearRegression().fit(X1, y)
           score =  reg.score(X1, y)
           scores_per_removed_dim[i] = score
           print(score)
           
    CTCF_only = X[:,1]
    sum_all_features = X.sum(axis=1)
    #CTCF_only  = X.sum(axis=1)
    idx_response = CTCF_only > 0
    temp1 = [j for j, x in enumerate(idx_response) if x]
    non_zero_f_l =  len(temp1)
    idx_LB = y > 1
    #idx_LB = y > 1
    temp2 = [j for j, x in enumerate(idx_LB) if x]
    lst3 = [value for value in temp1 if value in temp2] 
    high_B_num =  len(temp2)
    N = num_points
## A1 - total number for desired outcome in hypergeometric distribution 
    A1 = non_zero_f_l 
    n = high_B_num 
    x = len(lst3)
    print('Probability of Diffusion based Graph Laplacian')
    outcome_pmf = hypergeom_pmf(N, A1, n, x)
    hypergeom_plot(N, A1, n, x)
    print(outcome_pmf)
    print(x)
    print(A1)
    print(n)
    
  
def Compute_Betweeness_Centrality(A):
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    import random
    G =  nx.from_numpy_matrix(A)
    b = nx.edge_betweenness_centrality(G)
    values_b = b.values()
    keys_b = b.keys()
    A2 = np.zeros((np.shape(A)))                  
    for (k,v) in b.items():
        A2[k]=v
        
    y = A2.transpose()
    z_sub   = 10*np.add(A2, y)
    ## Divide by mean vertex degree value 
    BC = z_sub.sum(axis=0)
    BC = BC/np.mean(BC)
    
    return z_sub, BC

def Linear_Reg_test(X, BC, dim, num_points):
    import random
    from sklearn.linear_model import LinearRegression
    import numpy as np
     
    y1 = np.arange(num_points)
    np.random.shuffle(y1)
    y2 = BC[y1]
    scores_per_removed_dim = np.zeros(dim)
    y = BC
    for i in range(0,dim):
        if i==0: 
           X1 = X[:,i:]
        elif i==dim-1:
             X1 = X[:,0:dim-1]
        elif i>0 and i<dim-1:
             X11 = X[:,:i]
             X22 = X[:,i+1:]
             X1 = np.concatenate((X11, X22), axis=1)
    
        reg = LinearRegression().fit(X1, y)
        score =  reg.score(X1, y)
        scores_per_removed_dim[i] = score
        print(score)

    reg = LinearRegression().fit(X1, y2)
    score_rand =  reg.score(X1, y2)
    print('random')
    print(score_rand)

    return score, score_rand

def Hypergeometric_test_parameters(feature, BC, subset_size):
    import random
    import numpy as np
    
    N = BC.shape[0]
    idx_response = feature > 0
    temp1 = [j for j, x in enumerate(idx_response) if x]
    non_zero_f_l =  len(temp1)
    num_of_population = non_zero_f_l 
    idx_LB = np.argsort(BC)
    temp2 = idx_LB[N-subset_size:]
    #idx_LB = BC > 1.2 # 0.8
    #temp2 = [j for j, x in enumerate(idx_LB) if x]
    subset_size =  len(temp2)
    lst3 = [value for value in temp1 if value in temp2] 
    num_success = len(lst3)
    
    return num_success, num_of_population, subset_size 


def Extract_Hiccups(edges_new, start_point_idx, num_points, feature):
    import numpy as np
    import networkx as nx
    import scipy
    
    edges_first = np.concatenate(edges_new)
    idx_temp_first = edges_first >= start_point_idx
    idx_edges_first = [j for j, l in enumerate(idx_temp_first) if l]
    idx_temp_sec = edges_first <= (start_point_idx + num_points)
    idx_edges_sec = [j for j, l in enumerate(idx_temp_sec) if l]
    idx_edges_new = [value for value in idx_edges_first if value in idx_edges_sec]
    
    idx_first =  np.array(idx_edges_first)
    idx_sec   =  np.array(idx_edges_sec)
    ent_first     = edges_first[idx_edges_new]
    edges_ent_new = ent_first-start_point_idx
    edges_ent_new = np.unique(edges_ent_new)
    
    idx_response = feature > 0
    temp1 = [j for j, x in enumerate(idx_response) if x]
    non_zero_f_l =  len(temp1)
    lstedge = [value for value in temp1 if value in edges_ent_new] 
    num_success = len(lstedge)
    num_edges =  len(edges_ent_new)
    
    idx_response = feature > 0
    temp1 = [j for j, x in enumerate(idx_response) if x]
    non_zero_f_l =  len(temp1)
    num_of_population = non_zero_f_l 
    
    
    return edges_ent_new, num_edges, num_success, num_of_population

def Dim_Red(X,dimension_reduced):
    import numpy as np
    import networkx as nx
    import scipy
    
    u, s, vh = np.linalg.svd(X)
    u1 = u [0:dimension_reduced,:]
    X = u.transpose()
    X = X.transpose()
    mean = X.mean(0)
    var = X.std(0)
    X = (X - mean) / (var + 1e-6)
    return X

