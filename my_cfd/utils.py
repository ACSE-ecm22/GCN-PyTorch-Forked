import  torch
from    torch import nn
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import sys
import pickle as pkl
import torch.nn.functional as F
import networkx as nx
import pickle
import os
from sklearn.svm import SVC

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

def split_data(feat, num_nodes):
   
    train_idx, val_idx, test_idx = int(feat.size(0)*0.7), int(feat.size(0)*0.2), int(feat.size(0)*0.1),
    print("train Index:, ", train_idx)
    train_mask = index_to_mask(train_idx, num_nodes)
    val_mask = index_to_mask(val_idx, num_nodes)
    test_mask = index_to_mask(test_idx, num_nodes)
    return train_mask, val_mask, test_mask


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def masked_loss(out, label, mask):

    loss = F.cross_entropy(out, label, reduction='none')
    mask = mask.float()
    mask = mask / mask.mean()
    loss *= mask
    loss = loss.mean()
    return loss


def masked_acc(out, label, mask):
    # [node, f]
    pred = out.argmax(dim=1)
    correct = torch.eq(pred, label).float()
    mask = mask.float()
    mask = mask / mask.mean()
    correct *= mask
    acc = correct.mean()
    return acc



def sparse_dropout(x, rate, noise_shape):
    """

    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte()
    i = x._indices() # [2, 49216]
    v = x._values() # [49216]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1./ (1-rate))

    return out


def dot(x, y, sparse=False):
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.mm(x, y)

    return res



def generate_mask(features, missing_rate, missing_type):

    if missing_type == 'uniform':
        return generate_uniform_mask(features, missing_rate)
    if missing_type == 'bias':
        return generate_bias_mask(features, missing_rate)
    if missing_type == 'struct':
        return generate_struct_mask(features, missing_rate)
    raise ValueError("Missing type {0} is not defined".format(missing_type))


def generate_uniform_mask(features, missing_rate):
    """

    Parameters
    ----------
    features : torch.tensor
    missing_rate : float

    Returns
    -------
    mask : torch.tensor
        mask[i][j] is True if features[i][j] is missing.

    """
    mask = torch.rand(size=features.size())
    mask = mask <= missing_rate
    return mask


def generate_bias_mask(features, ratio, high=0.9, low=0.1):
    """

    Parameters
    ----------
    features: torch.Tensor
    ratio: float
    high: float
    low: float

    Returns
    -------
    mask : torch.tensor
        mask[i][j] is True if features[i][j] is missing.

    """
    node_ratio = (ratio - low) / (high - low)
    feat_mask = torch.rand(size=(1, features.size(1)))
    high, low = torch.tensor(high), torch.tensor(low)
    feat_threshold = torch.where(feat_mask < node_ratio, high, low)
    mask = torch.rand_like(features) < feat_threshold
    return mask


def generate_struct_mask(features, missing_rate):
    """

    Parameters
    ----------
    features : torch.tensor
    missing_rate : float

    Returns
    -------
    mask : torch.tensor
        mask[i][j] is True if features[i][j] is missing.

    """
    node_mask = torch.rand(size=(features.size(0), 1))
    mask = (node_mask <= missing_rate).repeat(1, features.size(1))
    return mask


def apply_mask(features, mask):
    """

    Parameters
    ----------
    features : torch.tensor
    mask : torch.tensor

    """
    features[mask] = float('nan')




def MMD(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)

def _mix_rbf_kernel(X, Y, sigma_list):
    assert(X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)

def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    return mmd2

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def new_load_data(dataset_str, norm_adj=True, generative_flag=False): # {'pubmed', 'citeseer', 'cora'}
    if dataset_str in ['cora', 'citeseer', 'pubmed']:
        """Load data."""
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("./data/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("./data/{}/ind.{}.test.index".format(dataset_str, dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = list(range(len(y)))
        idx_val = list(range(len(y), len(y)+500))

        # retutn normalized data
        labels = np.argmax(labels, 1)
        labels = torch.LongTensor(labels)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        if not generative_flag:
            features = normalize_features(features)
        if norm_adj:
            adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        indices = torch.LongTensor(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0))
        values = torch.FloatTensor(adj.tocoo().data)
        adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))

        features = torch.FloatTensor(np.array(features.todense()))
    elif dataset_str in ['steam']:
        freq_item_mat = pickle.load(open(os.path.join(os.getcwd(), 'data', dataset_str, 'freq_item_mat.pkl'), 'rb'))
        features = pickle.load(open(os.path.join(os.getcwd(), 'data', dataset_str, 'sp_fts.pkl'), 'rb'))

        if not generative_flag:
            features = normalize_features(features)

        features = torch.FloatTensor(features.todense())

        adj = freq_item_mat.copy()
        adj[adj < 10.0] = 0.0
        adj[adj >= 10.0] = 1.0
        indices = np.where(adj!=0.0)
        rows = indices[0]
        cols = indices[1]
        values = np.ones(shape=[len(rows)])
        adj = sp.coo_matrix((values, (rows, cols)), shape=[adj.shape[0], adj.shape[1]])
        if norm_adj:
            adj = normalize_adj(adj + sp.eye(adj.shape[0]))

        indices = torch.LongTensor(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0))
        values = torch.FloatTensor(adj.data)
        adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))

        labels = None
        idx_train = None
        idx_val = None
        idx_test = None

    else:
        print('cannot process this dataset !!!')
        raise Exception

    return adj, features, labels, idx_train, idx_val, idx_test

def load_generated_features(path):
    fts = pkl.load(open(path, 'rb'))
    norm_fts = normalize_features(fts)
    norm_fts = torch.FloatTensor(norm_fts)
    return norm_fts

def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)


    return adj, features, labels, idx_train, idx_val, idx_test

def eliminate_self_loops(A):
    """Remove self-loops from the adjacency matrix."""
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def normalize_adj(A):
    """Compute D^-1/2 * A * D^-1/2."""
    # Make sure that there are no self-loops
    A = eliminate_self_loops(A)
    D = np.ravel(A.sum(1))
    D[D == 0] = 1  # avoid division by 0 error
    D_sqrt = np.sqrt(D)
    return A / D_sqrt[:, None] / D_sqrt[None, :]


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def cal_accuracy(train_fts, train_lbls, test_fts, test_lbls):
    clf = SVC(gamma='auto')
    clf.fit(train_fts, train_lbls)

    preds_lbls = clf.predict(test_fts)
    acc = accuracy(preds_lbls, test_lbls)
    return acc

