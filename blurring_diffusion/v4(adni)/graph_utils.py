import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import pandas as pd
import os
from itertools import combinations, groupby
import random


# -------- Mask batch of node features with 0-1 flags tensor --------
def mask_x(x, flags):

    if flags is None:
        flags = torch.ones((x.shape[0], x.shape[1]), device=x.device)
    return x * flags[:,:,None]


# -------- Mask batch of adjacency matrices with 0-1 flags tensor --------
def mask_adjs(adjs, flags):
    """
    :param adjs:  B x N x N or B x C x N x N
    :param flags: B x N
    :return:
    """
    if flags is None:
        flags = torch.ones((adjs.shape[0], adjs.shape[-1]), device=adjs.device)

    if len(adjs.shape) == 4:
        flags = flags.unsqueeze(1)  # B x 1 x N
    adjs = adjs * flags.unsqueeze(-1)
    adjs = adjs * flags.unsqueeze(-2)
    return adjs


# -------- Create flags tensor from graph dataset --------
def node_flags(adj, eps=1e-5):

    flags = torch.abs(adj).sum(-1).gt(eps).to(dtype=torch.float32)

    if len(flags.shape)==3:
        flags = flags[:,0,:]
    return flags


# -------- Create initial node features --------
def init_features(init, adjs=None, nfeat=10):

    if init=='zeros':
        feature = torch.zeros((adjs.size(0), adjs.size(1), nfeat), dtype=torch.float32, device=adjs.device)
    elif init=='ones':
        feature = torch.ones((adjs.size(0), adjs.size(1), nfeat), dtype=torch.float32, device=adjs.device)
    elif init=='deg':
        feature = adjs.sum(dim=-1).to(torch.long)
        num_classes = nfeat
        try:
            feature = F.one_hot(feature, num_classes=num_classes).to(torch.float32)
        except:
            print(feature.max().item())
            raise NotImplementedError(f'max_feat_num mismatch')
    else:
        raise NotImplementedError(f'{init} not implemented')
    
    flags = node_flags(adjs)
    
    return mask_x(feature, flags)


# -------- Sample initial flags tensor from the training graph set --------
def init_flags(graph_list, config, batch_size=None):
    if batch_size is None:
        batch_size = config.data.batch_size
    max_node_num = config.data.max_node_num
    _, graph_tensor, _ = graphs_to_tensor(compute_L=False)
    idx = np.random.randint(0, len(graph_list), batch_size)
    flags = node_flags(graph_tensor[idx])

    return flags


# -------- Generate noise --------
def gen_noise(x, flags, sym=True):
    z = torch.randn_like(x)
    if sym:
        z = z.triu(1)
        z = z + z.transpose(-1,-2)
        z = mask_adjs(z, flags)
    else:
        z = mask_x(z, flags)
    return z


# -------- Quantize generated graphs -------
def quantize(adjs, thr=0.5):
    # adjs_min = adjs - adjs.min()
    # adjs_normalized = adjs_min / adjs_min.max()
    # adjs_ = torch.where(adjs_normalized < thr, torch.zeros_like(adjs_normalized), torch.ones_like(adjs_normalized))
    adjs_ = torch.where(adjs < thr, torch.zeros_like(adjs), torch.ones_like(adjs))
    return adjs_


# -------- Quantize generated molecules --------
# adjs: 32 x 9 x 9
def quantize_mol(adjs):                         
    if type(adjs).__name__ == 'Tensor':
        adjs = adjs.detach().cpu()
    else:
        adjs = torch.tensor(adjs)
    adjs[adjs >= 2.5] = 3
    adjs[torch.bitwise_and(adjs >= 1.5, adjs < 2.5)] = 2
    adjs[torch.bitwise_and(adjs >= 0.5, adjs < 1.5)] = 1
    adjs[adjs < 0.5] = 0
    return np.array(adjs.to(torch.int64))


def adjs_to_graphs(adjs, is_cuda=False):
    graph_list = []
    for adj in adjs:
        if is_cuda:
            adj = adj.detach().cpu().numpy()
        G = nx.from_numpy_array(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() < 1:
            G.add_node(1)
        graph_list.append(G)
    return graph_list


# -------- Check if the adjacency matrices are symmetric --------
def check_sym(adjs, print_val=False):
    sym_error = (adjs-adjs.transpose(-1,-2)).abs().sum([0,1,2])
    if not sym_error < 1e-2:
        raise ValueError(f'Not symmetric: {sym_error:.4e}')
    if print_val:
        print(f'{sym_error:.4e}')


# -------- Create higher order adjacency matrices --------
def pow_tensor(x, cnum):
    # x : B x N x N
    x_ = x.clone()
    xc = [x.unsqueeze(1)]
    for _ in range(cnum-1):
        x_ = torch.bmm(x_, x)
        xc.append(x_.unsqueeze(1))
    xc = torch.cat(xc, dim=1)

    return xc


# -------- Create padded adjacency matrices --------
def pad_adjs(ori_adj, node_number):
    a = ori_adj
    ori_len = a.shape[-1]
    if ori_len == node_number:
        return a
    if ori_len > node_number:
        raise ValueError(f'ori_len {ori_len} > node_number {node_number}')
    a = np.concatenate([a, np.zeros([ori_len, node_number - ori_len])], axis=-1)
    a = np.concatenate([a, np.zeros([node_number - ori_len, node_number])], axis=0)
    return a


def graphs_to_tensor(compute_L = True):
    node_list = []
    adjs_list = []
    label_list = []
    adjs_nl_list = []
    
    dt = pd.read_csv('./ADNI/data_fix/DT_thickness_labeling.csv')
    all_adjs = os.listdir('./ADNI/matrices2326/')
    
    for _ in range(len(all_adjs)):
        node = dt[all_adjs[_][11:] == dt['Subject']].iloc[:, 4:].to_numpy()
        label = dt[all_adjs[_][11:] == dt['Subject']].iloc[:, 2].to_numpy()
        adj = np.loadtxt(f'./ADNI/matrices2326/{all_adjs[_]}', dtype = int)
        
        node_tensor = torch.tensor(node.transpose(), dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        adjs_tensor = torch.tensor(adj, dtype=torch.float32)
        adjs_tensor = quantize(adjs_tensor)

        if label == 0:
            diag = torch.diag(adjs_tensor.sum(axis = -1))
            if torch.sum(diag != 0).item() == adj.shape[0]:
                adjs_nl_list.append(adjs_tensor)
                
        node_list.append(node_tensor)
        label_list.append(label_tensor)
        adjs_list.append(adjs_tensor)
    
    adjs_nl_tensor = torch.stack(adjs_nl_list)
    eigval_tensor, eigvec_tensor = compute_laplacian(adjs_nl_tensor.mean(0))
    
    node_tensor = torch.stack(node_list)
    label_tensor = torch.stack(label_list)
    adjs_tensor = torch.stack(adjs_list)

    del node_list
    del label_list
    del adjs_list
    
    # for g in graph_list:
    #     assert isinstance(g, nx.Graph)
    #     node_list = []
    #     for v, feature in g.nodes.data('feature'):
    #         node_list.append(v)

    #     adj = nx.to_numpy_array(g, nodelist=node_list)
    #     eigenvalue, eigenvector = compute_laplacian(adj)
    #     padded_adj = pad_adjs(adj, node_number=max_node_num)
    #     padded_eigval = pad_adjs(eigenvalue, node_number=max_node_num)
    #     padded_eigvec = pad_adjs(eigenvector, node_number=max_node_num)
    #     adjs_list.append(padded_adj)
    #     eigval_list.append(padded_eigval)
    #     eigvec_list.append(padded_eigvec)
        
    # del graph_list

    # adjs_np = np.asarray(adjs_list)
    # eigval_np = np.asarray(eigval_list)
    # eigvec_np = np.asarray(eigvec_list)
    # del adjs_list
    # del eigval_list
    # del eigvec_list

    # adjs_tensor = torch.tensor(adjs_np, dtype=torch.float32)
    # eigval_tensor = torch.tensor(eigval_np, dtype=torch.float32)
    # eigvec_tensor = torch.tensor(eigvec_np, dtype=torch.float32)
    # del adjs_np
    # del eigval_np
    # del eigvec_np

    return (node_tensor, adjs_tensor, label_tensor, eigval_tensor, eigvec_tensor) if compute_L else (node_tensor, adjs_tensor, label_tensor)


# def graphs_to_adj(graph, max_node_num):
#     max_node_num = max_node_num

#     assert isinstance(graph, nx.Graph)
#     node_list = []
#     for v, feature in graph.nodes.data('feature'):
#         node_list.append(v)

#     adj = nx.to_numpy_array(graph, nodelist=node_list)
#     padded_adj = pad_adjs(adj, node_number=max_node_num)

#     adj = torch.tensor(padded_adj, dtype=torch.float32)
#     del padded_adj

#     return adj


def node_feature_to_matrix(x):
    """
    :param x:  BS x N x F
    :return:
    x_pair: BS x N x N x 2F
    """
    x_b = x.unsqueeze(-2).expand(x.size(0), x.size(1), x.size(1), -1)  # BS x N x N x F
    x_pair = torch.cat([x_b, x_b.transpose(1, 2)], dim=-1)  # BS x N x N x 2F

    return x_pair

def random_graph(max_node_num, max_feat_num, p):
    nodes = [*range(1, max_node_num + 1, 1)]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    
    edges = combinations(nodes, 2)
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(max_node_num, create_using=G)
    
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
    
    while max(sorted(d for n, d in G.degree())) != (max_feat_num - 1):
        u, v = random.sample(nodes, 2)
        if random.random() < p:
            G.add_edge(u, v)

    return G

def compute_laplacian(x): 
    x = x.to('cuda')
    diag = torch.diag(x.sum(axis = -1))
    diag_inv_sqrt = torch.linalg.inv(diag)
    laplacian = diag - x
    
    normalized = torch.matmul(torch.sqrt(diag_inv_sqrt), torch.matmul(laplacian, torch.sqrt(diag_inv_sqrt)))

    eigenvalues, eigenvectors = torch.linalg.eig(normalized)
    eigenvalues = eigenvalues.real.detach().cpu()
    eigenvalues = eigenvalues[:, None] ** 2 + eigenvalues[None, :] ** 2
    eigenvectors = eigenvectors.real.detach().cpu()
    return eigenvalues, eigenvectors


# def gen_init_data(train_graph_list, batch_size):
#     rand_idx = np.random.randint(0, len(train_graph_list), batch_size)
#     graph_list = [train_graph_list[i] for i in rand_idx]
#     base_adjs, base_x = graphs_to_tensor(config, graph_list)
#     #base_adjs, base_x = base_adjs.to(config.dev), base_x.to(config.dev)
#     node_flags = base_adjs.sum(-1).gt(1e-5).to(dtype=torch.float32)

#     # Create a matrix with p=1/2 elements at all positions Aij where i and j not masked by node_flagij=0:
#     bernoulli_adj = torch.zeros(batch_size, max_node_number, max_node_number).to(config.dev)
#     for k, matrix in enumerate(base_adjs):
#         for i,row in enumerate(matrix):
#                 for j,col in enumerate(row):
#                     if 1/2 < node_flags[k][i] and 1/2 < node_flags[k][j]:
#                         bernoulli_adj[k,i,j] = 1/2            
#     noise_upper = torch.bernoulli(bernoulli_adj).triu(diagonal=1)
#     noise_lower = noise_upper.transpose(-1, -2)
#     initialmatrix = noise_lower + noise_upper
#     return initialmatrix, base_x, node_flags