from torch.utils.data import TensorDataset, DataLoader, random_split
from data.data_generators import load_dataset
from utils.graph_utils import init_features, graphs_to_tensor, adjs_to_graphs
import pandas as pd
import numpy as np
import torch
import os


def graphs_to_dataloader(config, get_graph_list = False):

    node_tensor, adjs_tensor, label_tensor, eigval_tensor, eigvec_tensor = graphs_to_tensor() # (graph list #, max node #, max node #)
   
    test_size = int(config.data.test_split * len(node_tensor))
    
    # ds = TensorDataset(node_tensor, adjs_tensor, label_tensor)
    train_ds = TensorDataset(node_tensor[test_size:], adjs_tensor[test_size:], label_tensor[test_size:], eigval_tensor[test_size:], eigvec_tensor[test_size:])
    test_ds = TensorDataset(node_tensor[:test_size], adjs_tensor[:test_size], label_tensor[:test_size], eigval_tensor[:test_size], eigvec_tensor[:test_size])
    # train_ds, test_ds = random_split(ds, [len(node_tensor) - test_size, test_size])

    if get_graph_list:
        train_adjs_list = []
        test_adjs_list = []
        for _ in range(len(train_ds)):
            train_adjs_list.append(train_ds[_][1])
        for _ in range(len(test_ds)):
            test_adjs_list.append(test_ds[_][1])
          
        train_graph_list = adjs_to_graphs(torch.stack(train_adjs_list).numpy())
        test_graph_list = adjs_to_graphs(torch.stack(test_adjs_list).numpy())
        
        return train_graph_list, test_graph_list
    
    train_dl = DataLoader(train_ds, batch_size=config.data.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=config.data.batch_size, shuffle=True)

    return train_dl, test_dl

def dataloader(config, get_graph_list=False):
    graph_list = load_dataset(data_dir=config.data.dir, file_name=config.data.data)

    test_size = int(config.data.test_split * len(graph_list))
    train_graph_list, test_graph_list = graph_list[test_size:], graph_list[:test_size]
    if get_graph_list:
        return train_graph_list, test_graph_list

    return graphs_to_dataloader(config, graph_list)
