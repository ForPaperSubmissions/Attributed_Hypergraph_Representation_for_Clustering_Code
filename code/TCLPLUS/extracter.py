import os.path as osp
from pathlib import Path
import json
import ast
import numpy as np
import matplotlib.pyplot as plt

path_parent = Path(__file__).parent.parent

method = "ours_Tricl"
dataset = "20news"
path = osp.join(path_parent, 'data', dataset, "result", method + "_quality_param.txt")

params_arr = []
measures_arr = []

f = open(path, "r")
for x in f:
    if x.startswith('{'):
        params = ast.literal_eval(x)
        params_arr.append(params)

    elif x.startswith("acc:"):
        measures = {}

        measures_str = x.split(";")
        for measure_str in measures_str:
            if measure_str.strip() == "":
                continue
            tmp = measure_str.split(":")
            measures[tmp[0]] = float(tmp[1])

        measures_arr.append(measures)
        
        
print(f'params_arr {len(params_arr)}')
print(f'measures_arr {len(measures_arr)}')

map_rate2idx = {0.1:0,0.3:1,0.5:2,0.7:3,0.9:4}
yticks = [0.1, 0.3, 0.5, 0.7, 0.9]
ylabel = "drop_dyadic_rate"

map_w2idx = {0.0625:0,0.125:1,0.25:2,0.5:3,1.0:4,2.0:5,4.0:6,8.0:7,16.0:8}
xticks = ['$2^{-4}$', '$2^{-3}$', '$2^{-2}$', '$2^{-1}$', '$2^{0}$', '$2^{1}$', '$2^{2}$', '$2^{3}$', '$2^{4}$']
xlabel = "w_d"

# for k in ['acc','nmi','ari','f1','jcc','balri','pur']:
for k in ['nmi','ari','f1','jcc','balri','pur']:
    arr = [[-1] * len(map_w2idx) for _ in range(len(map_rate2idx))]

    for i in range(len(measures_arr)):
        params = params_arr[i]
        measures = measures_arr[i]

        idx1 = map_rate2idx[params['drop_dyadic_rate']]
        idx2 = map_w2idx[params['w_d']]

        score = float(measures[k])
        
        arr[idx1][idx2] = score
    
    print(f'{len(arr)} {len(arr[0])}')

    plt.yticks(np.arange(len(yticks)), yticks)
    plt.ylabel(ylabel)

    plt.xticks(np.arange(len(xticks)), xticks)
    plt.xlabel(xlabel)

    plt.imshow(arr)
    plt.colorbar()
    plt.savefig(osp.join(path_parent, 'data', dataset, "result", "ours_Tricl", method + "_" + k + ".png"))
    plt.clf()

    