import numpy as np
import scipy.sparse as sp
import os, argparse
from sknetwork.clustering import Louvain
from sknetwork.linalg import normalize
from data import data
from scipy.sparse import diags
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
from metrics import clustering_metrics
import time, tracemalloc
import sparcifier
import psutil
import inspect

approx_knn = False
p = argparse.ArgumentParser(description='Set parameter')
p.add_argument('--dataset', type=str, default="coau_cora")
p.add_argument('--timer', type=bool, default=False)
p.add_argument('--knn_K', type=int, default=10)
p.add_argument('--alpha', type=float, default=0.2)
p.add_argument('--gamma', type=int, default=2)
p.add_argument('--tau', type=int, default=3)
args = p.parse_args()


# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

def run(dataset, tau):

    ###################### Setup ######################
    louvain = Louvain()

    f0 = args.alpha
    f1 = args.alpha * (1.0 - args.alpha)
    f2 = args.alpha * (1.0 - args.alpha)**2
    f3 = args.alpha * (1.0 - args.alpha)**3
    f4 = args.alpha * (1.0 - args.alpha)**4
    f5 = args.alpha * (1.0 - args.alpha)**5
    f6 = args.alpha * (1.0 - args.alpha)**6

    tracemalloc.start()
    
    print(f'AHRC. dataset {args.dataset} gamma {args.gamma} tau {args.tau}')

    peak_memory_before = psutil.virtual_memory().used

    # path to the dataset
    current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    path_to_data = os.path.join(current, 'data', dataset)
    
    ###################### Load Data ######################
    load_data = data.load(dataset)
    features = load_data['features']
    labels = load_data['labels']
    ground_truth = np.asarray(np.argmax(labels, axis=1)) if labels.ndim == 2 else labels
    hg_adj = load_data['adj']
    n = hg_adj.shape[1]

    ###################### Load or Compute ASM ######################
    knn_path = os.path.join(path_to_data, 'knn.npz')
    knn = None
    # check if file exist
    if os.path.isfile(knn_path):
        # print(f'read knn from file {knn_path}')
        knn = sp.load_npz(knn_path)
        knn.data = 1.0-knn.data
    else:
        if approx_knn:
            # print(f'compute apprx knn')
            import scann
            ftd = features.todense()
            if dataset.startswith('amazon'):
                searcher = scann.scann_ops_pybind.load_searcher('scann_amazon')
            else:
                knn_path = os.path.join(path_to_data, 'scann_magpm')
                print(f'knn_path {knn_path}')
                searcher = scann.scann_ops_pybind.load_searcher(knn_path)
            neighbors, distances = searcher.search_batched_parallel(ftd)
            del ftd
            knn = sp.csr_matrix(((distances.ravel()), neighbors.ravel(), np.arange(0, neighbors.size+1, neighbors.shape[1])), shape=(n, n))
            knn.setdiag(0.0)
            # save knn to npz file
            knn_path = os.path.join(path_to_data, 'knn_apprx.npz')
            sp.save_npz(knn_path, knn)
            # print("save apprx knn to file")
        else:
            # print("compute exact knn")
            knn = kneighbors_graph(features, args.knn_k, metric="cosine", mode="distance", n_jobs=16)
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
    P_topo = sparcifier.pack_tree_sum0(P_topo, n, tau)

    # perform random walk
    if args.gamma == 1:
        P_topo = f0 * diags(np.ones(n)) + f1 * P_topo
    elif args.gamma == 2:
        P_topo = f0 * diags(np.ones(n)) + f1 * P_topo + f2 * P_topo@P_topo
    elif args.gamma == 3:
        P_topo = f0 * diags(np.ones(n)) + f1 * P_topo + f2 * P_topo@P_topo + f3 * P_topo@P_topo@P_topo
    elif args.gamma == 4:
        P_topo = f0 * diags(np.ones(n)) + f1 * P_topo + f2 * P_topo@P_topo + f3 * P_topo@P_topo@P_topo + f4 * P_topo@P_topo@P_topo@P_topo
    elif args.gamma == 5:
        P_topo = f0 * diags(np.ones(n)) + f1 * P_topo + f2 * P_topo@P_topo + f3 * P_topo@P_topo@P_topo + f4 * P_topo@P_topo@P_topo@P_topo + f5 * P_topo@P_topo@P_topo@P_topo@P_topo
    elif args.gamma == 6:
        P_topo = f0 * diags(np.ones(n)) + f1 * P_topo + f2 * P_topo@P_topo + f3 * P_topo@P_topo@P_topo + f4 * P_topo@P_topo@P_topo@P_topo + f5 * P_topo@P_topo@P_topo@P_topo@P_topo + f6 * P_topo@P_topo@P_topo@P_topo@P_topo@P_topo

    # row normalization
    P_topo = normalize(P_topo, norm='l1')

    ###################### Integrate TSM and ASM ######################
    PP_sum = P_topo@P_attr

    ###################### transformation ######################
    # square root
    print("Square root transformation")
    pw = 1.0 / 2.0
    PP_sum.data = PP_sum.data ** pw

    ###################### Clustering ######################
    clusters = louvain.fit_predict(PP_sum)

    ###################### Statistics ######################
    tot_runningTime = time.time() - tot_start_time
    print("Running Time " + str(tot_runningTime))

    peak_memory_after = psutil.virtual_memory().used
    peak_mem_usage = (peak_memory_after - peak_memory_before) / (1024.0 ** 3)    
    print("Peak Memory usage: " + str(peak_mem_usage))
    
    # measure the clustering results
    cm = clustering_metrics(ground_truth, clusters)
    AHRC_metrics = cm.evaluationClusterModelFromLabel("AHRC", path_to_data)
    print("\n\n")

    return [str(tot_runningTime), AHRC_metrics]

timer_trial = 10

output = None
if args.timer:
    output = open("output_time.txt", "a")
else:
    output = open("output_metrics.txt", "a")

if args.timer:

    print("\n\n###################   " + args.dataset + "   ###################")

    runningTimes_AHRC = []
    
    for trial in range(timer_trial + 1):
        times = run(args.dataset, args.tau)
        runningTimes_AHRC.append(float(times[0]))

    # compute the average running time
    runningTimes_AHRC.sort()
    avg_runningTime_AHRC = sum(runningTimes_AHRC[:len(runningTimes_AHRC) - 1]) / (len(runningTimes_AHRC) - 1.0)
    print("Average running time AHRC " + f"{avg_runningTime_AHRC}")

    # save to file output.txt, remove the previous content
    output.write(f'dataset {args.dataset} gamma {args.gamma} tau {args.tau}\n')
    output.write("method: AHRC" + "\n")

    output.write("time\n")
    output.write(str(avg_runningTime_AHRC) + "\n")

    output.write("\n\n")
    output.flush()

else:

    print("\n\n###################   " + args.dataset + "   ###################")

    results = run(args.dataset, args.tau)

    # save to file output.txt, remove the previous content
    output.write(f'dataset {args.dataset} gamma {args.gamma} tau {args.tau}\n')
    output.write("method: AHRC" + "\n")

    output.write("time\n")
    output.write(str(results[0]) + "\n")
    output.write("cluster#\tnmi\tf1\tari\tjcc\tbalacc\tpurity\n")
    output.write(str(results[1]) + "\n")
    
    output.write("\n\n")
    output.flush()

output.close()