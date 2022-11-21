import scipy.sparse as sp
from scipy.sparse import linalg
import numpy as np
import torch


def normalize_adj(adj):
    for i in range(adj.shape[0]):
        adj[i, i] = 1
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def build_sparse_matrix(device, lap):
    shape = lap.shape
    i = torch.LongTensor(np.vstack((lap.row, lap.col)).astype(int))
    v = torch.FloatTensor(lap.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)


def get_cheb_polynomial(l_tilde, k):
    l_tilde = sp.coo_matrix(l_tilde)
    num = l_tilde.shape[0]
    cheb_polynomials = [sp.eye(num).tocoo(), l_tilde.copy()]
    for i in range(2, k + 1):
        cheb_i = (2 * l_tilde).dot(cheb_polynomials[i - 1]) - cheb_polynomials[i - 2]
        cheb_polynomials.append(cheb_i.tocoo())
    return cheb_polynomials


def get_supports_matrix(adj_mx, filter_type='laplacian', undirected=True):
    supports = []
    if filter_type == "laplacian":
        supports.append(calculate_scaled_laplacian(adj_mx, lambda_max=None, undirected=undirected))
    elif filter_type == "random_walk":
        supports.append(calculate_random_walk_matrix(adj_mx).T)
    elif filter_type == "dual_random_walk":
        supports.append(calculate_random_walk_matrix(adj_mx).T)
        supports.append(calculate_random_walk_matrix(adj_mx.T).T)
    else:
        supports.append(calculate_scaled_laplacian(adj_mx))
    return supports


def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    adj_mx = sp.coo_matrix(adj_mx)
    if undirected:
        bigger = adj_mx > adj_mx.T
        smaller = adj_mx < adj_mx.T
        notequall = adj_mx != adj_mx.T
        adj_mx = adj_mx - adj_mx.multiply(notequall) + adj_mx.multiply(bigger) + adj_mx.T.multiply(smaller)
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(lap, 1, which='LM')
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32).tocoo()
