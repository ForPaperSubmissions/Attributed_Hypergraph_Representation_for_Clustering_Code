import os.path as osp
from pathlib import Path
import json
import ast
import numpy as np
import matplotlib.pyplot as plt

path_parent = Path(__file__).parent.parent.parent

method = "GRACE(R)"
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

# drop_edge_rate_1 = {0.0:0,0.1:1,0.2:2,0.3:3,0.4:4}
# drop_edge_rate_2 = {0.0:0,0.1:1,0.2:2,0.3:3,0.4:4}

drop_edge_rate_1 = {0.0:0,0.1:1,0.2:2,0.3:3,0.4:4,0.5:5,0.6:6,0.7:7,0.8:8,0.9:9,1.0:10}
yticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ylabel = "drop_edge_rate"

drop_edge_rate_2 = {0.0:0,0.1:1,0.2:2,0.3:3,0.4:4,0.5:5,0.6:6,0.7:7,0.8:8,0.9:9,1.0:10}
xticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
xlabel = "drop_feature_rate"

# for k in ['acc','nmi','ari','f1','jcc','balri','pur']:
for k in ['nmi','ari','f1','jcc','balri','pur']:
    arr = [[-1] * len(drop_edge_rate_2) for _ in range(len(drop_edge_rate_1))]

    for i in range(len(measures_arr)):
        params = params_arr[i]
        measures = measures_arr[i]

        idx1 = drop_edge_rate_1[params['drop_edge_rate_1']]
        idx2 = drop_edge_rate_2[params['drop_feature_rate_1']]

        score = float(measures[k])
        
        arr[idx1][idx2] = score
    
    print(f'{len(arr)} {len(arr[0])}')

    plt.yticks(np.arange(len(yticks)), yticks)
    plt.ylabel(ylabel)

    plt.xticks(np.arange(len(xticks)), xticks)
    plt.xlabel(xlabel)

    plt.imshow(arr)
    plt.colorbar()
    plt.savefig(osp.join(path_parent, 'data', dataset, "result", "GRACE", method + "_" + k + ".png"))
    plt.clf()

    