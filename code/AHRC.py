import numpy as np
import scipy.sparse as sp
import os, argparse
from sknetwork.clustering import Louvain
from sknetwork.linalg import normalize
from data import data
from scipy.sparse import diags
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
import config
from metrics import clustering_metrics
import time, tracemalloc
import sparcifier

import psutil

datasets = []
datasets.append("coau_cora")
datasets.append("coci_cora")
datasets.append("coci_citeseer")

p = argparse.ArgumentParser(description='Set parameter')
p.add_argument('--dataset', type=str, default=None, help='dataset name (e.g.: cora/dblp for coauthorship, cora/citeseer for cocitation)')
p.add_argument('--alpha', type=float, default=0.2, help='mhc parameter')
p.add_argument('--knn_k', type=int, default=10, help='None')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--tau', type=int, default=3, help='None')
p.add_argument('--sparcify', type=str, default="symmetric_tree")
p.add_argument('--gamma', type=int, default=2)
args = p.parse_args()

sparcify = args.sparcify
gamma = args.gamma

# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

def run(dataset, tau):

    louvain = Louvain()

    f0 = config.alpha
    f1 = config.alpha * (1.0 - config.alpha)
    f2 = config.alpha * (1.0 - config.alpha)**2
    f3 = config.alpha * (1.0 - config.alpha)**3
    f4 = config.alpha * (1.0 - config.alpha)**4
    f5 = config.alpha * (1.0 - config.alpha)**5
    f6 = config.alpha * (1.0 - config.alpha)**6

    tracemalloc.start()

    config.dataset = dataset
    config.alpha = args.alpha
    config.knn_k = args.knn_k

    if config.knn_k == -1:
        config.knn_k = dataset['n'] - 1
    
    print(f'AHRC. dataset {config.dataset} gamma {gamma} tau {tau}')

    peak_memory_before = psutil.virtual_memory().used
    
    # load data
    load_data = data.load(config.dataset)
    features = load_data['features_sp']
    labels = load_data['labels']
    config.labels = np.asarray(np.argmax(labels, axis=1)) if labels.ndim == 2 else labels
    hg_adj = load_data['adj_sp']
    n = hg_adj.shape[1]
    config.hg_adj = hg_adj

    ###################### Load or Compute ASM ######################
    knn_path = os.path.join(config.prepath, 'knn.npz')
    knn = None
    # check if file exist
    if os.path.isfile(knn_path):
        print(f'read knn from file {knn_path}')
        knn = sp.load_npz(knn_path)
        knn.data = 1.0-knn.data
    else:
        if config.approx_knn:
            print(f'compute apprx knn')
            import scann
            ftd = features.todense()
            if config.dataset.startswith('amazon'):
                searcher = scann.scann_ops_pybind.load_searcher('scann_amazon')
            else:
                knn_path = os.path.join(config.prepath, 'scann_magpm')
                print(f'knn_path {knn_path}')
                searcher = scann.scann_ops_pybind.load_searcher(knn_path)
            neighbors, distances = searcher.search_batched_parallel(ftd)
            del ftd
            knn = sp.csr_matrix(((distances.ravel()), neighbors.ravel(), np.arange(0, neighbors.size+1, neighbors.shape[1])), shape=(n, n))
            knn.setdiag(0.0)
            # save knn to npz file
            knn_path = os.path.join(config.prepath, 'knn_apprx.npz')
            sp.save_npz(knn_path, knn)
            print("save apprx knn to file")
        else:
            print("compute exact knn")
            knn = kneighbors_graph(features, config.knn_k, metric="cosine", mode="distance", n_jobs=16)
            # save knn to npz file
            sp.save_npz(knn_path, knn)
            # print("save knn to file")
            knn.data = 1.0-knn.data

    ###################### Symmetric and row normalize ASM ######################
    # knn = knn + knn.T
    P_attr = normalize(knn + knn.T, norm='l1')

    tot_start_time = time.time()

    ###################### Compute TSM ######################

    # hypergraph random walk transition matrix
    P_topo = normalize(hg_adj.T, norm='l1', axis=1)@(normalize(hg_adj, norm='l1', axis=1))
    
    # perform spanning forest sparsification
    stime_spanningTree = time.time()
    P_topo = sparcifier.pack_tree_sum0(P_topo, n, tau)
    etime_spanningTree = time.time()

    # perform random walk
    if gamma == 1:
        P_topo = f0 * diags(np.ones(n)) + f1 * P_topo
    elif gamma == 2:
        P_topo = f0 * diags(np.ones(n)) + f1 * P_topo + f2 * P_topo@P_topo
    elif gamma == 3:
        P_topo = f0 * diags(np.ones(n)) + f1 * P_topo + f2 * P_topo@P_topo + f3 * P_topo@P_topo@P_topo
    elif gamma == 4:
        P_topo = f0 * diags(np.ones(n)) + f1 * P_topo + f2 * P_topo@P_topo + f3 * P_topo@P_topo@P_topo + f4 * P_topo@P_topo@P_topo@P_topo
    elif gamma == 5:
        P_topo = f0 * diags(np.ones(n)) + f1 * P_topo + f2 * P_topo@P_topo + f3 * P_topo@P_topo@P_topo + f4 * P_topo@P_topo@P_topo@P_topo + f5 * P_topo@P_topo@P_topo@P_topo@P_topo
    elif gamma == 6:
        P_topo = f0 * diags(np.ones(n)) + f1 * P_topo + f2 * P_topo@P_topo + f3 * P_topo@P_topo@P_topo + f4 * P_topo@P_topo@P_topo@P_topo + f5 * P_topo@P_topo@P_topo@P_topo@P_topo + f6 * P_topo@P_topo@P_topo@P_topo@P_topo@P_topo

    # row normalization
    P_topo = normalize(P_topo, norm='l1')

    ###################### Integrate TSM and ASM ######################
    PP_sum = P_topo@P_attr

    ###################### transformation ######################
    # square root
    # print("x = power(1/2)")
    pw = 1.0 / 2.0
    PP_sum.data = PP_sum.data ** pw

    # Linear
    # print("Linear")

    # exponental
    # print("exp(x) - 1 / exp(1) - 1")
    # PP_sum = PP_sum.expm1()
    # PP_sum.data = PP_sum.data / (np.exp(1) - 1)

    # Linear-over-Logarithm
    # print("Linear-over-Logarithm")
    # PP_sum.data = PP_sum.data / np.log2(1.0/PP_sum.data + 1)

    ###################### Clustering ######################
    stime_louvain = time.time()
    clusters = louvain.fit_predict(PP_sum)


    ###################### Statistics ######################
    tot_runningTime = time.time() - tot_start_time
    print("Running Time " + str(tot_runningTime))

    peak_memory_after = psutil.virtual_memory().used
    peak_mem_usage = (peak_memory_after - peak_memory_before) / (1024.0 ** 3)    
    print("Peak Memory usage: " + str(peak_mem_usage))
    
    # measure the clustering results
    cm = clustering_metrics(config.labels, clusters)
    MAHC_metrics = cm.evaluationClusterModelFromLabel("lv_A1")
    print("\n\n")

    return [str(tot_runningTime), MAHC_metrics]

calTime = False
time_trial = 10
min_tau = 3
max_tau = 3

output = None
if calTime:
    output = open("output_time.txt", "a")
else:
    output = open("output_metrics.txt", "a")

if calTime:

    for dataset in datasets:
        print("\n\n#########   " + dataset + "   ###################")
        
        tau = min_tau
        while tau <= max_tau:

            runningTimes_AHRC = []
            
            for trial in range(time_trial + 1):
                times = run(dataset, tau)
                runningTimes_AHRC.append(float(times[0]))

            # sort the values in arrays
            runningTimes_AHRC.sort()

            avg_runningTime_AHRC = sum(runningTimes_AHRC[:len(runningTimes_AHRC) - 1]) / (len(runningTimes_AHRC) - 1.0)
            print("average running time AHRC " + f"{avg_runningTime_AHRC}")

            # save to file output.txt, remove the previous content
            output.write(dataset + " tau " + str(tau) + "\n")
            output.write("AHRC" + "\n")

            output.write(str(avg_runningTime_AHRC) + "\n")

            output.write("\n\n")
            output.flush()

            tau += 1

else:

    for dataset in datasets:
        print("\n\n#########   " + dataset + "   ###################")

        tau = min_tau
        while tau <= max_tau:

            results = run(dataset, tau)

            # save to file output.txt, remove the previous content
            # [str(tot_runningTime), metrics, AHCKA_results[0], AHCKA_results[1]]
            output.write(dataset + " tau " + str(tau) + "\n")
            output.write("AHRC" + "\n")

            output.write(str(results[0]) + "\n")
            output.write(str(results[1]) + "\n")
            
            output.write("\n\n")
            output.flush()

            tau += 1

output.close()