import networkx as nx
import numpy as np
from numpy import linalg as LA
from scipy.linalg import eigh


def compute_norm_lap(A):
    import networkx as nx
    import numpy as np
    from numpy import linalg as LA
    from scipy.linalg import eigh
    from networkx.utils import not_implemented_for
   #  n, m = A.shape
   #  np.fill_diagonal(A, 0)
   #  diags = A.sum(axis=1)
   #  diags = np.sqrt(diags)
   #  diag_inv =1/diags 
   #  D = np.diagflat(diag_inv)
   # # diag=_sqrt[np.isinf(diag_sqrt)] = 0
   #  I = np.identity(m)     
   #  A_norm = np.matmul(A, D)
   #  A_norm = np.matmul(D, A_norm)
   #  L_norm = I - A_norm 
    G =  nx.from_numpy_matrix(A)
    A1 = nx.normalized_laplacian_matrix(G)
    L_norm = A1.todense()
    return L_norm


def compute_basis_lap(L, num_eig):
    bd, B =  eigh(L, subset_by_index=[0, num_eig-1])
    return bd, B


def compute_basis_Sylvester(A, L):
    import networkx as nx
    import numpy as np
    from numpy import linalg as LA
    from scipy.linalg import eigh
    
    I = np.identity(m)     
    x = Sy_Embeddings(A, L, I)
    bd, B =  eigh(x)
    return bd, B


def WKS(B,bd,num_descriptors):
    import networkx as nx
    import numpy as np
    from numpy import linalg as LA
    from scipy.linalg import eigh
    w = 7
    numEigenfunctions = bd.shape[0]
    Bt = transpose.B
    temp1 = np.matmul(B, B)
    temp2 = np.matmul(Bt, temp1)
    #BB = Bt * (B.^2)
    absoluteEigenvalues = abs(bd)
    l = len(absoluteEigenvalues)-1
    emax = np.log(absoluteEigenvalues)[l]
    emin = np.log(absoluteEigenvalues)[1]   
    emin = emin + 2*s
    emax = emax - 2*s
    es = np.linspace(emin, emax, num_descriptors)
    T1 = np.matlib.repmat(np.log(absoluteEigenvalues), 1, num_descriptors)
    T2 = np.matlib.repmat(es, numEigenfunctions,1)
    T  = np.exp( -numpy.multiply(T1-T2, T1-T2)/(2*s*s))
    wks = temp2*T;
    wks = B * wks;