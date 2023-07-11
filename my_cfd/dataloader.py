import  numpy as np
import  pickle as pkl
import  networkx as nx
import  scipy.sparse as sp
import torch
import  sys
from utils import sparse_mx_to_torch_sparse_tensor, split_data, normalize_adj





# def new_load_data(dataset_str, norm_adj=True, generative_flag=False): # {'pubmed', 'citeseer', 'cora'}
    

#     features = np.load("data_store/feature/regularWave_buoy_1.4.vtk_HIGH_features.npy").tolil()

#     # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
#     adj = nx.adjacency_matrix(sp.load_npz("data_store/adj/regularWave_buoy_1.4.vtk_HIGH_adj.npz"))


#     # labels = np.vstack((ally, ty))
#     # labels[test_idx_reorder, :] = labels[test_idx_range, :]

#     # idx_test = test_idx_range.tolist()
#     # idx_train = list(range(len(y)))
#     # idx_val = list(range(len(y), len(y)+500))

#     # retutn normalized data
#     # labels = np.argmax(labels, 1)
#     # labels = torch.LongTensor(labels)
#     # idx_train = torch.LongTensor(idx_train)
#     # idx_val = torch.LongTensor(idx_val)
#     # idx_test = torch.LongTensor(idx_test)

#     if not generative_flag:
#         features = normalize_features(features)
#     if norm_adj:
#         adj = normalize_adj(adj + sp.eye(adj.shape[0]))
#     indices = torch.LongTensor(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0))
#     values = torch.FloatTensor(adj.tocoo().data)
#     adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))

#     features = torch.FloatTensor(np.array(features.todense()))

#     if not generative_flag:
#         features = normalize_features(features)

#     features = torch.FloatTensor(features.todense())

#     adj = freq_item_mat.copy()
#     adj[adj < 10.0] = 0.0
#     adj[adj >= 10.0] = 1.0
#     indices = np.where(adj!=0.0)
#     rows = indices[0]
#     cols = indices[1]
#     values = np.ones(shape=[len(rows)])
#     adj = sp.coo_matrix((values, (rows, cols)), shape=[adj.shape[0], adj.shape[1]])
#     if norm_adj:
#         adj = normalize_adj(adj + sp.eye(adj.shape[0]))

#     indices = torch.LongTensor(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0))
#     values = torch.FloatTensor(adj.data)
#     adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))

#     labels = None
#     idx_train = None
#     idx_val = None
#     idx_test = None

#     return adj, features, labels, idx_train, idx_val, idx_test



class Data:
    def __init__(self):
        self.adj = sp.load_npz("data_store/adj/regularWave_buoy_1.4.vtk_HIGH_adj.npz")
        self.features = np.load("data_store/feature/regularWave_buoy_1.4.vtk_HIGH_features.npy")
        n_rows = 5000
        self.features = self.features[:n_rows, :]
        self.adj = self.adj[:n_rows, :n_rows]
        self.features = normalize_features(self.features)       
        self.features = torch.Tensor(self.features)
        self.num_features = self.features.size(1)

    def to(self, device):
        self.features = self.features.to(device)


class NodeClsData(Data):
    def __init__(self, norm_adj=True):
        super(NodeClsData, self).__init__()

        #train_mask, val_mask, test_mask = split_data(self.features, self.features.size(0))
        # self.train_mask = train_mask
        # self.val_mask = val_mask
        # self.test_mask = test_mask

        self.num_nodes = self.features.shape[0]

        idx_test = list(range(int(self.num_nodes * 0.2)))

        idx_train = list(range(int(self.num_nodes * 0.2), int(self.num_nodes * 0.9)))

        idx_val = list(range(int(self.num_nodes * 0.9), int(self.num_nodes * 0.1)))

        self.idx_train = torch.LongTensor(idx_train)
        self.idx_val = torch.LongTensor(idx_val)
        self.idx_test = torch.LongTensor(idx_test)

        if norm_adj == True:
            print("normalised adj")
            self.adj = normalize_adj(self.adj + sp.eye(self.adj.shape[0]))
        else:
            self.adj = self.adj

        

        self.adj = sparse_mx_to_torch_sparse_tensor(self.adj)
        #self.adj = torch.FloatTensor(np.array(self.adj.todense()))

    def to(self, device):
        """

        Parameters
        ----------
        device: string
            cpu or cuda

        """
        super().to(device)
        self.adj = self.adj.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)


def sample_mask(idx, l):
    """
    Create mask.
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data():

    # reducing for memory limitations
    n_rows = 10000
    features = np.load("data_store/feature/regularWave_buoy_1.4.vtk_HIGH_features.npy")
    adj = sp.load_npz("data_store/adj/regularWave_buoy_1.4.vtk_HIGH_adj.npz")

    train_mask, val_mask, test_mask = split_data(features, features.shape[0])

    features = features[:n_rows]
    adj = adj[:n_rows, :n_rows]


    features = torch.Tensor(features)
    adj = sparse_mx_to_torch_sparse_tensor(adj)


    return (adj, features)


def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx




