import argparse
import os.path as osp
import random
import yaml
from yaml import SafeLoader
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GCNConv

from model import Encoder, Model, drop_feature
from eval import label_classification

from loader import DatasetLoader
from evaluation import kmeans
import numpy as np

import os, psutil

# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

def train(model: Model, x, edge_index):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_edge(edge_index, p=p_e)[0]
    edge_index_2 = dropout_edge(edge_index, p=p_e)[0]
    x_1 = drop_feature(x, p_a)
    x_2 = drop_feature(x, p_a)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()

def fix_seed(seed):
    # pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # torch.set_deterministic_debug_mode(0)

def test(model: Model, x, edge_index, y, final=False):
    model.eval()
    z = model(x, edge_index)

    label_classification(z, y, ratio=0.1)

def clustering_eval(model: Model, x, edge_index, y, final=False):

    inference_start_time = time.time()
    model.eval()
    z = model(x, edge_index)
    inference_time = time.time() - inference_start_time

    trials = 5
    avg_clustering_time = 0
    nmis = []
    aris = []
    f1s = []
    jccs = []
    balris = []
    purs = []
    for i in range(trials):

        clutering_time, nmi, ari, f1, jcc, balri, pur = kmeans(z, y, i, args.dataset)
        avg_clustering_time += clutering_time

        nmis.append(nmi)
        aris.append(ari)
        f1s.append(f1)
        jccs.append(jcc)
        balris.append(balri)
        purs.append(pur)

    avg_clustering_time /= trials
    total_clustering_time = inference_time + avg_clustering_time

    return total_clustering_time, nmis, aris, f1s, jccs, balris, purs

def print_measure(save, dataset_dir, dataset, model_type, params, accs, nmis, aris, f1s, jccs, balris, purs, times):

    output = dataset + ";" + model_type + "\n"
    output += f'{params}' + "\n"

    accs = np.array(accs).reshape(-1, 5)
    accs_mean = list(np.mean(accs, axis=0))
    accs_std = list(np.std(accs, axis=0))
    output += f'acc:{accs_mean[2]:.2f}:{accs_std[2]:.2f}' + ";"

    nmis = np.array(nmis).reshape(-1, 5)
    nmis_mean = list(np.mean(nmis, axis=0))
    nmis_std = list(np.std(nmis, axis=0))
    output += f'nmi:{nmis_mean[2]:.2f}:{nmis_std[2]:.2f}' + ";"

    aris = np.array(aris).reshape(-1, 5)
    aris_mean = list(np.mean(aris, axis=0))
    aris_std = list(np.std(aris, axis=0))
    output += f'ari:{aris_mean[2]:.2f}:{aris_std[2]:.2f}' + ";"

    f1s = np.array(f1s).reshape(-1, 5)
    f1s_mean = list(np.mean(f1s, axis=0))
    f1s_std = list(np.std(f1s, axis=0))
    output += f'f1:{f1s_mean[2]:.2f}:{f1s_std[2]:.2f}' + ";"

    jccs = np.array(jccs).reshape(-1, 5)
    jccs_mean = list(np.mean(jccs, axis=0))
    jccs_std = list(np.std(jccs, axis=0))
    output += f'jcc:{jccs_mean[2]:.2f}:{jccs_std[2]:.2f}' + ";"

    balris = np.array(balris).reshape(-1, 5)
    balris_mean = list(np.mean(balris, axis=0))
    balris_std = list(np.std(balris, axis=0))
    output += f'balri:{balris_mean[2]:.2f}:{balris_std[2]:.2f}' + ";"

    purs = np.array(purs).reshape(-1, 5)
    purs_mean = list(np.mean(purs, axis=0))
    purs_std = list(np.std(purs, axis=0))
    output += f'pur:{purs_mean[2]:.2f}:{purs_std[2]:.2f}' + ";"

    times_mean = 0
    for time in times:
        times_mean += time
    times_mean /= len(times)
    output += f'time:{times_mean:.5f}' + ";"
    
    print(output)

    if save:
        path = osp.join(dataset_dir, 'results', 'GRCPLUS.txt')

        file = open(path, "a")
        file.write(output + "\n")
        file.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coau_cora')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--p_e', type=float, default=-1.0)
    parser.add_argument('--p_a', type=float, default=-1.0)
    parser.add_argument('--lr', type=float, default = -1.0)
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    params = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    
    if args.p_e >= 0.0:
        params['p_e'] = args.p_e
    if args.p_a >= 0.0:
        params['p_a'] = args.p_a
    if args.lr >= 0.0:
        params['lr'] = args.lr

    print(f'dataset {args.dataset}')
    print(params)

    lr = params['lr']
    num_hidden = params['num_hidden']
    num_proj_hidden = params['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[params['activation']]
    base_model = ({'GCNConv': GCNConv})[params['base_model']]
    num_layers = params['num_layers']
    p_a = params['p_a']
    p_e = params['p_e']
    tau = params['tau']
    num_epochs = params['num_epochs']
    weight_decay = params['weight_decay']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = DatasetLoader().load(args.dataset).to(device)

    accs, nmis, aris, f1s, jccs, balris, purs, times = [], [], [], [], [], [], [], []
    
    for seed in range(args.num_seeds):
        
        print(f'seed {seed}')
        fix_seed(seed)

        data_num_features = int(data.features.shape[1])
        data_x = data.features
        data_edge_index = data.dyadicedge_index
        data_y = data.labels

        encoder = Encoder(data_num_features, num_hidden, activation,
                        base_model=base_model, k=num_layers).to(device)
        model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay)

        start_trainning_time = time.time()
        for epoch in range(1, num_epochs + 1):
            loss = train(model, data_x, data_edge_index)
        trainning_time = time.time() - start_trainning_time

        clutering_time, nmi, ari, f1, jcc, balri, pur = clustering_eval(model, data_x, data_edge_index, data_y, final=True)
        
        total_running_time = trainning_time + clutering_time 
        print(f'total_running_time {total_running_time}')

        accs.append(nmi)
        nmis.append(nmi)
        aris.append(ari)
        f1s.append(f1)
        jccs.append(jcc)
        balris.append(balri)
        purs.append(pur)
        times.append(total_running_time)

    print_measure(True, data.dataset_dir, args.dataset, "GRCPLUS", params, accs, nmis, aris, f1s, jccs, balris, purs, times)
    print()
