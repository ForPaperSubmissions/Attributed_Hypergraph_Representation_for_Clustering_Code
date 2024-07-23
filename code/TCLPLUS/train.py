import argparse
import random
import os.path as osp

import yaml
from tqdm import tqdm
import numpy as np
import torch
import time

from loader import DatasetLoader
from models import HyperEncoder, TCLPLUS
from utils import drop_features, drop_incidence
from evaluation import linear_evaluation, kmeans

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

def print_measure(save, dataset_dir, dataset, model_type, params, accs, nmis, aris, f1s, jccs, balris, purs, times):

    output = dataset + ";" + model_type + "\n"
    output += f'{params}' + "\n"

    accs = np.array(accs).reshape(-1, 3)
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
        path = osp.join(dataset_dir, 'results', 'TCLPLUS_quality.txt')

        file = open(path, "a")
        file.write(output + "\n")
        file.close()

def train(model_type, num_negs, seed):
    features, hyperedge_index, dyadicedge_index = data.features, data.hyperedge_index, data.dyadicedge_index
    num_nodes, num_hyperedges = data.num_nodes, data.num_hyperedges

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Hypergraph Augmentation
    hyperedge_index1 = drop_incidence(hyperedge_index, params['drop_incidence_rate'], seed)
    hyperedge_index2 = drop_incidence(hyperedge_index, params['drop_incidence_rate'], seed)
    x1 = drop_features(features, params['drop_feature_rate'], seed)
    x2 = drop_features(features, params['drop_feature_rate'], seed)

    # Dyadic Graph Augmentation
    dyadicedge_index1 = drop_incidence(dyadicedge_index, params['p_d'], seed)
    dyadicedge_index2 = drop_incidence(dyadicedge_index, params['p_d'], seed)

    # Encoder
    n1, e1, dyadic_n1 = model(x1, hyperedge_index1, dyadicedge_index1, num_nodes, num_hyperedges, len(dyadicedge_index1[0]))
    n2, e2, dyadic_n2 = model(x2, hyperedge_index2, dyadicedge_index2, num_nodes, num_hyperedges, len(dyadicedge_index2[0]))

    # Projection Head
    n1, n2 = model.node_projection(n1), model.node_projection(n2)
    e1, e2 = model.edge_projection(e1), model.edge_projection(e2)
    dyadic_n1, dyadic_n2 = model.dyadic_node_projection(dyadic_n1), model.dyadic_node_projection(dyadic_n2)

    loss_n = model.node_level_loss(n1, n2, params['tau_n'], batch_size=params['batch_size_1'], num_negs=num_negs)
    loss_n += params['w_s'] * model.node_level_loss(dyadic_n1, dyadic_n2, params['tau_n'], batch_size=params['batch_size_1'], num_negs=num_negs)

    loss = loss_n
    loss.backward()
    optimizer.step()

    return loss.item()

def node_classification_eval(num_splits=20):
    model.eval()
    n, _, _ = model(data.features, data.hyperedge_index, data.dyadicedge_index)

    if data.name == 'pubmed':
        lr = 0.005
        max_epoch = 300
    elif data.name == 'cora' or data.name == 'citeseer':
        lr = 0.005
        max_epoch = 100
    elif data.name == 'Mushroom':
        lr = 0.01
        max_epoch = 200
    else:
        lr = 0.01
        max_epoch = 100

    accs = []
    for i in range(num_splits):
        masks = data.generate_random_split(seed=i)
        accs.append(linear_evaluation(n, data.labels, masks, lr=lr, max_epoch=max_epoch))
    return accs 

def clustering_eval():

    inference_start_time = time.time()
    model.eval()
    n, _, _ = model(data.features, data.hyperedge_index, data.dyadicedge_index)
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

        clutering_time, nmi, ari, f1, jcc, balri, pur = kmeans(n, data.labels, i, args.dataset)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TCL+.')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--model_type', type=str, default='TCLPLUS')
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--p_d', type=float, default = -1.0)
    parser.add_argument('--lr', type=float, default = -1.0)
    parser.add_argument('--w_s', type=float, default = -1.0)
    args = parser.parse_args()

    params = yaml.safe_load(open('config.yaml'))[args.dataset]

    if args.p_d >= 0.0:
        params['p_d'] = args.p_d
    if args.lr >= 0.0:
        params['lr'] = args.lr
    if args.w_s >= 0.0:
        params['w_s'] = args.w_s
    
    print(f'TCLPLUS {args.dataset} {args.model_type} {params}')

    data = DatasetLoader().load(args.dataset).to(args.device)

    accs, nmis, aris, f1s, jccs, balris, purs, times, memories = [], [], [], [], [], [], [], [], []
    for seed in range(args.num_seeds):
        fix_seed(seed)
        encoder = HyperEncoder(data.features.shape[1], params['hid_dim'], params['hid_dim'], params['num_layers'])
        model = TCLPLUS(encoder, params['proj_dim']).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

        start_trainning_time = time.time()
        for epoch in tqdm(range(1, params['epochs'] + 1)):
            loss = train(args.model_type, num_negs=None, seed=seed)
        trainning_time = time.time() - start_trainning_time

        acc = node_classification_eval()
        clutering_time, nmi, ari, f1, jcc, balri, pur = clustering_eval()

        total_running_time = trainning_time + clutering_time 
        print(f'total_running_time {total_running_time}')

        accs.append(acc)
        nmis.append(nmi)
        aris.append(ari)
        f1s.append(f1)
        jccs.append(jcc)
        balris.append(balri)
        purs.append(pur)
        times.append(total_running_time)

    print_measure(True, data.dataset_dir, args.dataset, args.model_type, params, accs, nmis, aris, f1s, jccs, balris, purs, times)
    print()