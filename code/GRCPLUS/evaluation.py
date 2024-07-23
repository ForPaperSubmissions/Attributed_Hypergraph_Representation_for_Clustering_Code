import torch
import torch.nn.functional as F
from torch import Tensor

from logreg import LogReg
import time
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np

ks = {
  "coau_cora": 8,
  "coci_cora": 11
}

def masked_accuracy(logits: Tensor, labels: Tensor):
    if len(logits) == 0:
        return 0
    pred = torch.argmax(logits, dim=1)
    acc = pred.eq(labels).sum() / len(logits) * 100
    return acc.item()


def accuracy(logits: Tensor, labels: Tensor, masks: list[Tensor]):
    accs = []
    for mask in masks:
        acc = masked_accuracy(logits[mask], labels[mask])
        accs.append(acc)
    return accs

def binomialCoe(m):
    if m == 0 or m == 1:
        return 1
    else:
        return m * (m - 1) / 2
        
def clusteringAcc_pairwise(Y_true, Y_predict):
    num_groundTruth_cluster = len(np.unique(Y_true))
    num_pred_cluster = len(np.unique(Y_predict))
    num_node = len(Y_true)
    # print("num_groundTruth_cluster " + str(num_groundTruth_cluster) + " num_pred_cluster " + str(num_pred_cluster))
    # print("num_node " + str(num_node))

    # create a linked list for each ground truth and discovered cluster
    groundTruth_cluster_list = []
    for i in range(num_groundTruth_cluster):
        groundTruth_cluster_list.append([])

    pred_cluster_list = []
    for i in range(num_pred_cluster):
        pred_cluster_list.append([])

    for i in range(num_node):
        groundTruth_cluster_list[Y_true[i]].append(i)
        pred_cluster_list[Y_predict[i]].append(i)

    # calculation
    N = binomialCoe(num_node)

    TPFP = 0
    TP = 0
    FP = 0

    # for each discovered cluster
    for i in range(num_pred_cluster):

        # counterArr[i] is the intersection of c1 with g_i
        counterArr = []
        for j in range(num_groundTruth_cluster):
            counterArr.append(0)

        clusterSize_Discover = 0
        # for each element in c1
        for vID in pred_cluster_list[i]:
            clusterSize_Discover += 1
            # find its ground truth cluster
            counterArr[Y_true[vID]] += 1

        # print(str(pred_cluster_list[i]) + " " + str(sum(counterArr)) + " " + str(clusterSize_Discover))
        TPFP += binomialCoe(clusterSize_Discover)

        # for each ground truth cluster
        for j in range(num_groundTruth_cluster):
            intersect = counterArr[j]
            if intersect < 1:
                continue
            # print("\t" + str(intersect))
            TP += binomialCoe(intersect)
        # print(str(TPFP) + " " + str(TP))

    FP = TPFP - TP
    # print("TPFP " + str(TPFP) + " TP " + str(TP) + " FP " + str(FP))
    
    TPFN = 0
    # for each ground truth cluster
    for i in range(num_groundTruth_cluster):
        clusterSize_GroundTruth = len(groundTruth_cluster_list[i])
        TPFN += binomialCoe(clusterSize_GroundTruth)
    
    FN = TPFN - TP
    FPTN = N - TPFN
    TN = FPTN - FP

    ri = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 0
    if ((precision + recall) == 0):
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    ari = 2 * ((TP * TN) - (FP * FN))
    ari = ari / (((TP + FN) * (FN + TN)) + ((TP + FP) * (FP + TN)))
    jcc = TP / (TP + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    balri = (sensitivity + specificity) / 2.0

    return num_pred_cluster, ri, f1, precision, recall, ari, jcc, balri
    
def purity(Y_true, Y_predict):
    num_groundTruth_cluster = len(np.unique(Y_true))
    num_pred_cluster = len(np.unique(Y_predict))
    num_node = len(Y_true)

    # create a linked list for each ground truth and discovered cluster
    groundTruth_cluster_list = []
    for i in range(num_groundTruth_cluster):
        groundTruth_cluster_list.append([])

    pred_cluster_list = []
    for i in range(num_pred_cluster):
        pred_cluster_list.append([])

    for i in range(num_node):
        groundTruth_cluster_list[Y_true[i]].append(i)
        pred_cluster_list[Y_predict[i]].append(i)

    N = 0
    purity_sort = []

    # for each discovered cluster
    for i in range(num_pred_cluster):

        # counterArr[i] is the intersection of c1 with g_i
        counterArr = [0]*num_groundTruth_cluster

        # for each element in c1
        for vID in pred_cluster_list[i]:
            N += 1
            # find its ground truth cluster
            counterArr[Y_true[vID]] += 1
        
        # find the maximum overlap
        maxOverlap = 0
        for j in range(num_groundTruth_cluster):
            if counterArr[j] > maxOverlap:
                maxOverlap = counterArr[j]

        purity_sort.append(maxOverlap)

    left = 1.0 / N
    right = 0
    purity = 0

    # print("num_groundTruth_cluster " + str(num_groundTruth_cluster) + " num_pred_cluster " + str(num_pred_cluster))
    if num_groundTruth_cluster <= num_pred_cluster:
        # sort purity_sort in descending order
        purity_sort.sort(reverse=True)

        # for each ground truth cluster
        for i in range(num_groundTruth_cluster):
            right += purity_sort[i]
        purity = left * right

    else:
        # for each discovered cluster
        for i in range(num_pred_cluster):  
            right += purity_sort[i]
        purity = left * right

    return purity

def kmeans(z, labels, seed, dataset):
    z = z.detach()
    X, num_classes = z.cpu().numpy(), int(labels.max()) + 1

    if len(ks) > 0:
        if dataset in ks:
            num_classes = ks[dataset]
            print(f'manually set cluster number {num_classes}', end = ' ')
            
    start_time = time.time()
    kmeans = KMeans(n_clusters=num_classes, random_state=seed, n_init=5).fit(X)
    clutering_time = time.time() - start_time

    Y_predict = kmeans.labels_
    Y = labels.detach().cpu().numpy()

    nmi = metrics.normalized_mutual_info_score(Y, Y_predict)
    ari = metrics.adjusted_rand_score(Y, Y_predict)
    num_pred_cluster, ri, f1, precision, recall, ari_pair, jcc, balri = clusteringAcc_pairwise(Y, Y_predict)
    pur = purity(Y, Y_predict)

    return clutering_time, nmi, ari, f1, jcc, balri, pur

def linear_evaluation(z, labels, masks, lr=0.01, max_epoch=100):
    z = z.detach()
    hid_dim, num_classes = z.shape[1], int(labels.max()) + 1

    classifier = LogReg(hid_dim, num_classes).to(z.device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.0)

    for epoch in range(1, max_epoch + 1):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)

        logits = classifier(z[masks[0]])
        loss = F.cross_entropy(logits, labels[masks[0]])
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        classifier.eval()
        logits = classifier(z)
        accs = accuracy(logits, labels, masks)

    return accs
