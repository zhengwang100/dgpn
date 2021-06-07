import os
import numpy as np
import scipy.sparse as sp
import torch
import random
import torch.nn.functional as F
from collections import defaultdict

def get_adj_raw_feat(G):
    features = row_normalize(G)
    features = torch.from_numpy(features)
    print(features.shape, type(features))
    return features

def load_data_set(dataset):
    filepath = 'datasets'
    label_file = os.path.join(filepath, '{}/group.txt'.format(dataset))
    edge_file = os.path.join(filepath,'{}/graph.txt'.format(dataset))
    feature_file = os.path.join(filepath, '{}/feature.txt'.format(dataset))
    csd_file = os.path.join(filepath, 'csd_files/{}_text_csds.txt'.format(dataset)) # or 'CSD/{}_label_csds.txt'

    idx, labellist = read_node_label(label_file)
    G = read_graph_as_matrix(nodeids=idx, edge_file= edge_file)
    features = np.genfromtxt(feature_file, dtype=np.float)[:, 1:]
    if( ('M10-M' in dataset) is False ):
        features = row_normalize(features)
    csd_matrix = get_csd_matrix(csd_file)

    return idx, labellist, G, torch.FloatTensor(features), csd_matrix

def get_csd_matrix(csd_file):
    csdmatrix = np.loadtxt(csd_file) # [c, csd_dim]
    csdmatrix = torch.FloatTensor(csdmatrix)
    csdmatrix = F.normalize(csdmatrix, p=2, dim=1)
    return csdmatrix

def dot_sim(x, y):
    # Inner product similarity
    ip_sim = torch.mm(x, y)
    return ip_sim

def get_data_split(c_train, c_val, idx, labellist):
    '''Input: 
        idx: list[n, 1]
        labellist: list[n, string]
    Return:
            train_list: [num_train_samples, 1]
            val_list: [num_val_samples, 1]
            test_list: [num_test_samples, 1]
            total_class: num_class
    '''
    label_list_dict = defaultdict(list)
    for x, labels in zip(idx, labellist):
        for y in labels: 
            label_list_dict[int(y)].append(int(x))

    train_list = []; val_list = []; test_list = []
    for i in label_list_dict.keys():
        #print(i, len(label_list_dict[i]))
        if i < c_train: 
            train_list = train_list + label_list_dict[i]
        elif c_train <= i < (c_train+c_val):
            val_list = val_list + label_list_dict[i]
        else: test_list = test_list + label_list_dict[i]
    #print(len(train_list), len(val_list), len(test_list))
    return train_list, test_list, val_list 

def get_acc(pred, label, c_train, c_val, model):
    # assume the c_train, c_val, c_test are ranked according to their c_ids
    mypred = torch.ones(pred.shape)*float('-inf')
    if(model == 'train'):
        mypred[:, :c_train] = pred[:, :c_train]
    elif model == 'val':
        mypred[:, c_train: c_train+c_val] = pred[:, c_train: c_train+c_val]
    elif model == 'test':
        mypred[:, c_train+c_val: ] = pred[:, c_train+c_val: ]
    return get_acc_basic(mypred, label)

def get_acc_basic(predict, label):
    predict = torch.argmax(predict, axis=1)
    acc = (label.cpu()==predict)
    result = acc.cpu().sum().numpy()
    #print('Train true/false acc:', result/len(acc))
    return result/len(acc)

# ------------------------------------- 
def read_node_label(filename):
    #print(os.getcwd())
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split()
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y

def symmetrize(adj):
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    ####adj = np.maximum(adj, adj.transpose() )
    return adj.todense()

def read_graph_as_matrix(nodeids, edge_file):
    ''' Read a symmetric adjacency matrix from a file
        Input: nodeids: [1,2,3,4,...]
        Return: the sparse adjacency matrix
    '''
    idx = np.array(nodeids, dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(edge_file, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(idx), len(idx)),
                        dtype=np.float32)
    print('origial input G', type(adj), sp.coo_matrix.count_nonzero(adj))
    # build symmetric adjacency matrix
    adj = symmetrize(adj)
    #print('symmetrice adj type', type(adj))
    return adj

def symmetric_normalize_adj(adj):
    """Symmetrically normalize adjacency matrix: D^{-1/2}AD^{-1/2}"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
    
def row_normalize(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def use_cuda():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    return device