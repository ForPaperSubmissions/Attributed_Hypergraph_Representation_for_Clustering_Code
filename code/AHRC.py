import numpy as np
import scipy.sparse as sp
import os, sys, argparse
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

import psutil, resource, gc

datasets = []
datasets.append("coci_patent_C13")
datasets.append("coci_wiki")
datasets.append("coau_cora")
datasets.append("coci_cora")
datasets.append("coci_citeseer")
datasets.append("20news")
datasets.append("coci_pubmed")
datasets.append("coau_dblp")
datasets.append("NTU2012")
# datasets.append("magpm")

p = argparse.ArgumentParser(description='Set parameter')
p.add_argument('--dataset', type=str, default=None, help='dataset name (e.g.: cora/dblp for coauthorship, cora/citeseer for cocitation)')
p.add_argument('--tmax', type=int, default=200, help='t_max parameter')
p.add_argument('--seeds', type=int, default=0, help='seed for randomness')
p.add_argument('--alpha', type=float, default=0.2, help='mhc parameter')
p.add_argument('--beta', type=float, default=0.5, help='weight of knn random walk')
p.add_argument('--metric', type=bool, default=False, help='calculate additional metrics: modularity')
p.add_argument('--weighted_p', type=int, default=0, help='use transition matrix p weighted by attribute similarity')
p.add_argument('--verbose', action='store_true', help='print verbose logs')
p.add_argument('--scale', action='store_true', help='use configurations for large-scale data')
p.add_argument('--interval', type=int, default=5, help='interval between cluster predictions during orthogonal iterations')
p.add_argument('--knn_k', type=int, default=10, help='None')
p.add_argument('--w_topo', type=int, default=1, help='None')
p.add_argument('--w_attr', type=int, default=1, help='None')
p.add_argument('--isAND', type=int, default=1, help='None')
p.add_argument('--num_trees', type=int, default=3, help='None')

p.add_argument('--sparcify', type=str, default="symmetric_tree")
p.add_argument('--seed', type=int, default=0)
p.add_argument('--gamma', type=int, default=2)
args = p.parse_args()

sparcify = args.sparcify
gamma = args.gamma

# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

def run(dataset, num_trees):

    louvain = Louvain()

    f0 = config.alpha
    f1 = config.alpha * (1.0 - config.alpha)
    f2 = config.alpha * (1.0 - config.alpha)**2
    f3 = config.alpha * (1.0 - config.alpha)**3
    f4 = config.alpha * (1.0 - config.alpha)**4
    f5 = config.alpha * (1.0 - config.alpha)**5
    f6 = config.alpha * (1.0 - config.alpha)**6

    # starting the monitoring
    tracemalloc.start()

    config.dataset = dataset
    config.alpha = args.alpha
    config.knn_k = args.knn_k
    config.w_topo = args.w_topo
    config.w_attr = args.w_attr
    config.isAND = args.isAND

    if config.knn_k == -1:
        config.knn_k = dataset['n'] - 1
    
    print("AHRC. dataset " + str(config.dataset) + " knn_k " + str(config.knn_k) + " w_topo " + str(config.w_topo) + " w_attr " + str(config.w_attr))
    print(f'gamma = {gamma}')

    # gc.collect()
    peak_memory_before = psutil.virtual_memory().used
    mem_before = process_memory()
    
    # load data
    load_data = data.load(config.dataset)
    features = load_data['features_sp']
    labels = load_data['labels']
    config.labels = np.asarray(np.argmax(labels, axis=1)) if labels.ndim == 2 else labels
    hg_adj = load_data['adj_sp']

    # preprocessing
    n = hg_adj.shape[1]
    config.hg_adj = hg_adj

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

    # knn = knn + knn.T
    P_attr = normalize(knn + knn.T, norm='l1')

    tot_start_time = time.time()

    P_topo = normalize(hg_adj.T, norm='l1', axis=1)@(normalize(hg_adj, norm='l1', axis=1))
    
    if sparcify == "tree":

        print(f'sparcify: tree {num_trees}')

        stime_spanningTree = time.time()

        P_topo = sparcifier.tree(P_topo, n, num_trees)
        
        etime_spanningTree = time.time()
        print("time for spanning tree " + str(etime_spanningTree - stime_spanningTree))

    elif sparcify == "symmetric_tree":
        print(f'sparcify: symmetric_tree {num_trees}')

        stime_spanningTree = time.time()
        P_topo = sparcifier.pack_tree_sum0(P_topo, n, num_trees)
        etime_spanningTree = time.time()
        print("time for spanning tree " + str(etime_spanningTree - stime_spanningTree))
        
    elif sparcify == "random":
        print("random")
        P_topo = sparcifier.independent_sampling(P_topo, n, num_trees, args.seed)

    else:
        print("sparcifier not defined")
        exit()

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

    P_topo = normalize(P_topo, norm='l1')

    PP_sum = P_topo@P_attr

    # arr = PP_sum.toarray()
    # print(n)
    # print(np.sum(P_topo, axis=1))
    # print(np.sum(P_attr, axis=1))
    # print(np.sum(arr, axis=1))

    # Linear
    # print("Linear")
    # print(PP)

    # x = x^1/2
    # print("x = power(1/2)")
    # pw = 1.0 / 2.0
    # PP_sum.data = PP_sum.data ** pw
    # print(PP)

    # x = x^1/3
    # print("x = power(1/3)")
    # pw = 1.0 / 3.0
    # PP_sum.data = PP_sum.data ** pw
    # print(PP)

    # power of 2
    # PP_sum.data = PP_sum.data ** 2
    # PP = PP_sum
    # print(PP)

    # exp(x) - 1 / exp(1) - 1
    # print("exp(x) - 1 / exp(1) - 1")
    # PP_sum = PP_sum.expm1()
    # PP_sum.data = PP_sum.data / (np.exp(1) - 1)
    # PP = PP_sum
    # print(PP)

    # Linear-over-Logarithm
    print("Linear-over-Logarithm")
    PP_sum.data = PP_sum.data / np.log2(1.0/PP_sum.data + 1)
    # PP = PP_sum
    # print(PP)

    # run Louvain
    stime_louvain = time.time()
    clusters = louvain.fit_predict(PP_sum)
    print("time for louvain " + str(time.time() - stime_louvain))

    tot_runningTime = time.time() - tot_start_time
    print("running time " + str(tot_runningTime))

    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)
    print("meory " + str(peak_memory))
    peak_memory_after = psutil.virtual_memory().used
    peak_mem_usage = (peak_memory_after - peak_memory_before) / (1024.0 ** 3)    
    mem_after = process_memory()
    mem_usage = (mem_after - mem_before) / (1024.0 ** 3)
    print("Memory usage: " + str(mem_usage))
    print("Peak Memory usage: " + str(peak_mem_usage))
    
    # measure the clustering results
    cm = clustering_metrics(config.labels, clusters)
    MAHC_metrics = cm.evaluationClusterModelFromLabel("lv_A1")
    print("\n\n")

    gc.collect()

    AHCKA_runningTime = 0.0
    AHCKA_metrics = None
    if False:
        sys.path.append('baselines/AHCKA/')
        import AHCKA
        AHCKA_results = AHCKA.run(config.dataset, len(np.unique(clusters)))
        AHCKA_runningTime = AHCKA_results[0]
        AHCKA_metrics = AHCKA_results[1]
        print("\n\n")

    # gc.collect()

    GRAC_runningTime = 0.0
    GRAC_metrics = None
    if False:
        sys.path.append('baselines/GRAC/')
        import GRAC
        GRAC_results = GRAC.run(config.dataset, len(np.unique(clusters)))
        GRAC_runningTime = GRAC_results[0]
        GRAC_metrics = GRAC_results[1]
        print("\n\n")
    
    # gc.collect()

    JNMF_runningTime = 0.0
    JNMF_metrics = None
    if False:
        sys.path.append('baselines/JNMF/')
        import JNMF
        JNMF_results = JNMF.run(config.dataset, len(np.unique(clusters)))
        JNMF_runningTime = JNMF_results[0]
        JNMF_metrics = JNMF_results[1]
        print("\n\n")

    # gc.collect()

    GNMFA_runningTime = 0.0
    GNMFA_metrics = None
    if False:
        sys.path.append('baselines/GNMF/')
        import GNMF
        GNMFA_results = GNMF.run(config.dataset, len(np.unique(clusters)), 'HyperAdj')
        GNMFA_runningTime = GNMFA_results[0]
        GNMFA_metrics = GNMFA_results[1]
        print("\n\n")

    # gc.collect()

    GNMFC_runningTime = 0.0
    GNMFC_metrics = None
    if False:
        sys.path.append('baselines/GNMF/')
        import GNMF
        GNMFC_results = GNMF.run(config.dataset, len(np.unique(clusters)), 'clique')
        GNMFC_runningTime = GNMFC_results[0]
        GNMFC_metrics = GNMFC_results[1]
        print("\n\n")

    # gc.collect()

    GNMFL_runningTime = 0.0
    GNMFL_metrics = None
    if False:
        sys.path.append('baselines/GNMF/')
        import GNMF
        GNMFL_results = GNMF.run(config.dataset, len(np.unique(clusters)), 'HyperNcut')
        GNMFL_runningTime = GNMFL_results[0]
        GNMFL_metrics = GNMFL_results[1]
        print("\n\n")

    return [str(tot_runningTime), MAHC_metrics, AHCKA_runningTime, AHCKA_metrics, GRAC_runningTime, GRAC_metrics, JNMF_runningTime, JNMF_metrics, GNMFA_runningTime, GNMFA_metrics, GNMFC_runningTime, GNMFC_metrics, GNMFL_runningTime, GNMFL_metrics]

if args.dataset is None:
    calTime = False
    time_trial = 10
    min_num_trees = 3
    max_num_trees = 3
    
    output = None
    if calTime:
        output = open("output_time.txt", "a")
    else:
        output = open("output_metrics.txt", "a")

    if calTime:

        for dataset in datasets:
            print("\n\n#########   " + dataset + "   ###################")
            
            num_trees = min_num_trees
            while num_trees <= max_num_trees:

                runningTimes_AHCKA = []
                runningTimes_GRAC = []
                runningTimes_JNMF = []
                runningTimes_GNMFA = []
                runningTimes_GNMFC = []
                runningTimes_GNMFL = []
                runningTimes_AHRC = []
                
                for trial in range(time_trial + 1):
                    times = run(dataset, num_trees)
                    runningTimes_AHCKA.append(float(times[2]))
                    runningTimes_GRAC.append(float(times[4]))
                    runningTimes_JNMF.append(float(times[6]))
                    runningTimes_GNMFA.append(float(times[8]))
                    runningTimes_GNMFC.append(float(times[10]))
                    runningTimes_GNMFL.append(float(times[12]))
                    runningTimes_AHRC.append(float(times[0]))

                # sort the values in arrays
                runningTimes_AHCKA.sort()
                runningTimes_GRAC.sort()
                runningTimes_JNMF.sort()
                runningTimes_GNMFA.sort()
                runningTimes_GNMFC.sort()
                runningTimes_GNMFL.sort()
                runningTimes_AHRC.sort()

                avg_runningTime_AHCKA = sum(runningTimes_AHCKA[:len(runningTimes_AHCKA) - 1]) / (len(runningTimes_AHCKA) - 1.0)
                avg_runningTime_GRAC = sum(runningTimes_GRAC[:len(runningTimes_GRAC) - 1]) / (len(runningTimes_GRAC) - 1.0)
                avg_runningTime_JNMF = sum(runningTimes_JNMF[:len(runningTimes_JNMF) - 1]) / (len(runningTimes_JNMF) - 1.0)
                avg_runningTime_GNMFA = sum(runningTimes_GNMFA[:len(runningTimes_GNMFA) - 1]) / (len(runningTimes_GNMFA) - 1.0)
                avg_runningTime_GNMFC = sum(runningTimes_GNMFC[:len(runningTimes_GNMFC) - 1]) / (len(runningTimes_GNMFC) - 1.0)
                avg_runningTime_GNMFL = sum(runningTimes_GNMFL[:len(runningTimes_GNMFL) - 1]) / (len(runningTimes_GNMFL) - 1.0)
                avg_runningTime_AHRC = sum(runningTimes_AHRC[:len(runningTimes_AHRC) - 1]) / (len(runningTimes_AHRC) - 1.0)
                
                print("average running time AHCKA " + f"{avg_runningTime_AHCKA}")
                print("average running time GRAC " + f"{avg_runningTime_GRAC}")
                print("average running time JNMF " + f"{avg_runningTime_JNMF}")
                print("average running time GNMFA " + f"{avg_runningTime_GNMFA}")
                print("average running time GNMFC " + f"{avg_runningTime_GNMFC}")
                print("average running time GNMFL " + f"{avg_runningTime_GNMFL}")
                print("average running time AHRC " + f"{avg_runningTime_AHRC}")

                # save to file output.txt, remove the previous content
                output.write(dataset + " tau " + str(num_trees) + "\n")
                output.write("AHCKA" + "\n")
                output.write("GRAC" + "\n")
                output.write("JNMF" + "\n")
                output.write("GNMFA" + "\n")
                output.write("GNMFC" + "\n")
                output.write("GNMFL" + "\n")
                output.write("AHRC" + "\n")

                output.write(str(avg_runningTime_AHCKA) + "\n")
                output.write(str(avg_runningTime_GRAC) + "\n")
                output.write(str(avg_runningTime_JNMF) + "\n")
                output.write(str(avg_runningTime_GNMFA) + "\n")
                output.write(str(avg_runningTime_GNMFC) + "\n")
                output.write(str(avg_runningTime_GNMFL) + "\n")
                output.write(str(avg_runningTime_AHRC) + "\n")
                
                output.write("\n\n")
                output.flush()

                num_trees += 1

    else:

        for dataset in datasets:
            print("\n\n#########   " + dataset + "   ###################")

            num_trees = min_num_trees
            while num_trees <= max_num_trees:

                if args.sparcify == "random":

                    for seed in range(5):

                        print(f'tree {num_trees} seed {seed}')

                        args.seed = seed
                        results = run(dataset, num_trees)

                        # save to file output.txt, remove the previous content
                        # [str(tot_runningTime), metrics, AHCKA_results[0], AHCKA_results[1]]
                        output.write(dataset + " tau " + str(num_trees) + " seed " + str(seed) + "\n")
                        output.write("AHCKA" + "\n")
                        output.write("GRAC" + "\n")
                        output.write("JNMF" + "\n")
                        output.write("GNMFA" + "\n")
                        output.write("GNMFC" + "\n")
                        output.write("GNMFL" + "\n")
                        output.write("AHRC" + "\n")

                        output.write(str(results[2]) + "\n")
                        output.write(str(results[4]) + "\n")
                        output.write(str(results[6]) + "\n")
                        output.write(str(results[8]) + "\n")
                        output.write(str(results[10]) + "\n")
                        output.write(str(results[12]) + "\n")
                        output.write(str(results[0]) + "\n")
                        
                        output.write(str(results[3]) + "\n")
                        output.write(str(results[5]) + "\n")
                        output.write(str(results[7]) + "\n")
                        output.write(str(results[9]) + "\n")
                        output.write(str(results[11]) + "\n")
                        output.write(str(results[13]) + "\n")
                        output.write(str(results[1]) + "\n")
                        
                        output.write("\n\n")
                        output.flush()

                else:
                    print(num_trees)
                    results = run(dataset, num_trees)

                    # save to file output.txt, remove the previous content
                    # [str(tot_runningTime), metrics, AHCKA_results[0], AHCKA_results[1]]
                    output.write(dataset + " tau " + str(num_trees) + "\n")
                    output.write("AHCKA" + "\n")
                    output.write("GRAC" + "\n")
                    output.write("JNMF" + "\n")
                    output.write("GNMFA" + "\n")
                    output.write("GNMFC" + "\n")
                    output.write("GNMFL" + "\n")
                    output.write("AHRC" + "\n")

                    output.write(str(results[2]) + "\n")
                    output.write(str(results[4]) + "\n")
                    output.write(str(results[6]) + "\n")
                    output.write(str(results[8]) + "\n")
                    output.write(str(results[10]) + "\n")
                    output.write(str(results[12]) + "\n")
                    output.write(str(results[0]) + "\n")
                    
                    output.write(str(results[3]) + "\n")
                    output.write(str(results[5]) + "\n")
                    output.write(str(results[7]) + "\n")
                    output.write(str(results[9]) + "\n")
                    output.write(str(results[11]) + "\n")
                    output.write(str(results[13]) + "\n")
                    output.write(str(results[1]) + "\n")
                    
                    output.write("\n\n")
                    output.flush()

                num_trees += 1

    output.close()
else:

    results = run(args.dataset, args.num_trees)
    
    output = open("output.txt", "w")
    output.write(args.dataset + " tau " + str(args.num_trees) + "\n")
    output.write("AHCKA" + "\n")
    output.write("GRAC" + "\n")
    output.write("JNMF" + "\n")
    output.write("GNMFA" + "\n")
    output.write("GNMFC" + "\n")
    output.write("GNMFL" + "\n")
    output.write("AHRC" + "\n")

    output.write(str(results[2]) + "\n")
    output.write(str(results[4]) + "\n")
    output.write(str(results[6]) + "\n")
    output.write(str(results[8]) + "\n")
    output.write(str(results[10]) + "\n")
    output.write(str(results[12]) + "\n")
    output.write(str(results[0]) + "\n")
    
    output.write(str(results[3]) + "\n")
    output.write(str(results[5]) + "\n")
    output.write(str(results[7]) + "\n")
    output.write(str(results[9]) + "\n")
    output.write(str(results[11]) + "\n")
    output.write(str(results[13]) + "\n")
    output.write(str(results[1]) + "\n")
    
    output.write("\n\n")
    output.close()
