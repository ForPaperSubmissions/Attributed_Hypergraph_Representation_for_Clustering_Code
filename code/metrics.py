import numpy as np
from sklearn import metrics
import config

class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def investigate(self):
        num_groundTruth_cluster = len(np.unique(self.true_label))
        num_pred_cluster = len(np.unique(self.pred_label))
        num_node = len(self.true_label)

        # create a linked list for each ground truth and discovered cluster
        groundTruth_cluster_list = []
        for i in range(num_groundTruth_cluster):
            groundTruth_cluster_list.append([])

        pred_cluster_list = []
        for i in range(num_pred_cluster):
            pred_cluster_list.append([])

        for i in range(num_node):
            groundTruth_cluster_list[self.true_label[i]].append(i)
            pred_cluster_list[self.pred_label[i]].append(i)

        adj = config.hg_adj.T
        # for each hyperedge
        for i in range(0):
            # for each node
            row = config.hg_adj[i].toarray()[0]
            for j in range(len(row)):
                if row[j] == 1:
                    print(str(j) + "(" + str(adj[j].count_nonzero()) + ") " + str(self.true_label[j]) + " " + str(self.pred_label[j]))
            print("")

    def binomialCoe(self, m):
        if m == 0 or m == 1:
            return 1
        else:
            return m * (m - 1) / 2

    def clusteringAcc_pairwise(self):
        num_groundTruth_cluster = len(np.unique(self.true_label))
        num_pred_cluster = len(np.unique(self.pred_label))
        num_node = len(self.true_label)
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
            groundTruth_cluster_list[self.true_label[i]].append(i)
            pred_cluster_list[self.pred_label[i]].append(i)

        # calculation
        N = self.binomialCoe(num_node)

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
                counterArr[self.true_label[vID]] += 1

            # print(str(pred_cluster_list[i]) + " " + str(sum(counterArr)) + " " + str(clusterSize_Discover))
            TPFP += self.binomialCoe(clusterSize_Discover)

            # for each ground truth cluster
            for j in range(num_groundTruth_cluster):
                intersect = counterArr[j]
                if intersect < 1:
                    continue
                # print("\t" + str(intersect))
                TP += self.binomialCoe(intersect)
            # print(str(TPFP) + " " + str(TP))

        FP = TPFP - TP
        # print("TPFP " + str(TPFP) + " TP " + str(TP) + " FP " + str(FP))
        
        TPFN = 0
        # for each ground truth cluster
        for i in range(num_groundTruth_cluster):
            clusterSize_GroundTruth = len(groundTruth_cluster_list[i])
            TPFN += self.binomialCoe(clusterSize_GroundTruth)
        
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
    
    def purity(self):
        num_groundTruth_cluster = len(np.unique(self.true_label))
        num_pred_cluster = len(np.unique(self.pred_label))
        num_node = len(self.true_label)

        # create a linked list for each ground truth and discovered cluster
        groundTruth_cluster_list = []
        for i in range(num_groundTruth_cluster):
            groundTruth_cluster_list.append([])

        pred_cluster_list = []
        for i in range(num_pred_cluster):
            pred_cluster_list.append([])

        for i in range(num_node):
            groundTruth_cluster_list[self.true_label[i]].append(i)
            pred_cluster_list[self.pred_label[i]].append(i)

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
                counterArr[self.true_label[vID]] += 1
            
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

    def evaluationClusterModelFromLabel(self, method):
        
        save = True

        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        # acc, f1, pre, rc = self.clusteringAcc()
        num_pred_cluster, ri, f1, precision, recall, ari, jcc, balri = self.clusteringAcc_pairwise()
        purity = self.purity()

        # print
        print("|c| ri nmi f1 precision adjscore recall ari jcc balri purity")
        print(f"{num_pred_cluster}\t{ri:.5f}\t{nmi:.5f}\t{f1:.5f}\t{adjscore:.5f}\t{ari:.5f}\t{jcc:.5f}\t{balri:.5f}\t{purity:.5f}")

        # save results to a txt file
        if save:
            # write to a txt file
            path = config.prepath + "/metrics/" + method + ".txt"
            f = open(path, "a")
            f.write(method + "\n")
            f.write(f"{ri:.5f}\t{nmi:.5f}\t{f1:.5f}\t{adjscore:.5f}\t{ari:.5f}\t{jcc:.5f}\t{balri:.5f}\t{purity:.5f}" + "\n")
            f.write("\n")
            f.close()

        return f"{num_pred_cluster}\t{ri:.5f}\t{nmi:.5f}\t{f1:.5f}\t{adjscore:.5f}\t{ari:.5f}\t{jcc:.5f}\t{balri:.5f}\t{purity:.5f}"