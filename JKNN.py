import numpy as np
import pandas as pd
import random
import sys
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import time
import math


def kmor(X: np.array, k: int, y: float = 2.5, nc0: float = 0.1, max_iteration: int = 100, gamma: float = 10 ** -6):
    """K-means clustering with outlier removal
    Parameters
    ----------
    X
        Your data.
    k
        Number of clusters.
    y
        Parameter for outlier detection. Increase this to make outlier removal subtle.
    nc0
        Maximum percentage of your data that can be assigned to outlier cluster.
    max_iteration
        Maximum number of iterations.
    gamma
        Used to check the convergence.
    Returns
    -------
    numpy.array
        Numpy array that contains the assigned cluster of each data point (0 to k, the cluster k is the outlier
        cluster)
    """
    n = X.shape[0]
    n0 = int(math.ceil(nc0 * n))
    # print('max of outlier = ',n0)
    Z = X[np.random.choice(n, k)]

    def calculate_dd(U, Z):
        # print('data - center: \n',X - Z[U])
        return np.linalg.norm(X - Z[U], axis=1) ** 2

    def calculate_D(outliers, dd):
        factor = y / (n - outliers.size)
        # print('factor = ',factor)
        # print(dd)
        # print("dd(sum) = ",np.sum(dd) )
        if outliers.size < 1:
            return factor * np.sum(dd)
        return factor * np.sum(np.delete(dd, outliers))

    def calculate_U(X):
        def closest(p):
            return np.argmin(np.linalg.norm(Z - p, axis=1))

        return np.apply_along_axis(closest, 1, X)

    outliers = np.array([])
    U = calculate_U(X)
    s = 0
    p = 0
    while True:
        # Update U (Theorem 1)
        dd = calculate_dd(U, Z)
        D = calculate_D(outliers, dd)
        dd2 = dd[dd > D]
        outliers = np.arange(n)[dd > D][dd2.argsort()[::-1]]
        # print(Z)
        # print('D = ' , D)
        U = calculate_U(X)
        # print(U)
        # print(dd)
        # print(outliers)
        outliers = outliers[:n0]

        # Update Z (Theorem 3)
        is_outlier = np.isin(U, outliers)

        def mean_group(i):
            x = X[np.logical_and(U == i, ~is_outlier)]
            # Empty group
            if x.size == 0:
                x = X[np.random.choice(n, 1)]
            return x.mean(axis=0)

        Z = np.array([mean_group(i) for i in range(k)])
        # Update P
        dd = calculate_dd(U, Z)
        D = calculate_D(outliers, dd)
        if outliers.size == 0:
            p1 = np.sum(dd)
        else:
            p1 = np.sum(dd[~outliers]) + D * outliers.size
        # Exit condition
        s += 1
        if abs(p1 - p) < gamma or s >= max_iteration:
            break
        p = p1
    print("s:", s, "p:", p)
    U[outliers] = k
    temp = (U == k)  # | (dist < q1 - iqr * w)
    # print(temp)

    o = X[temp]
    n = X[~temp]
    # return U
    return o , n


# One-class J-K nearest neighbor algorithm
class JKNN:

    def __init__(self, j=2, k=2, theta=1 , OD = "default",NN = 'default'):  # 
        self.j = j
        self.k = k
        self.theta = theta
        self.OD = OD
        self.NN = NN
        self.nn = NearestNeighbors(n_neighbors=max(self.j, self.k))
        self.x = None

    def set(self, j, k, theta=1):
        self.j = j
        self.k = k
        self.theta = theta
        # self.nn = NearestNeighbors(n_neighbors=max(self.j, self.k))

    def outlierdetection(self, origin_data, g):  # 
        center = (np.sum(origin_data, axis=0)) / len(origin_data)
        dist = np.linalg.norm(origin_data - center, axis=1)
        dist_after_sort = np.sort(dist, axis=0)
        q3 = np.quantile(dist_after_sort, 0.75)
        q1 = np.quantile(dist_after_sort, 0.25)
        iqr = q3 - q1

        # print(dist_after_sort, q3, q1)

        o = None  # outlier
        n = None  # normal

        #   if dist > q3 + iqr*w or dist < q1 - iqr*w
        #   =>  this point is outlier
        w = 1.5  #

        while (1):
            temp = (dist > q3 + iqr * w)  # | (dist < q1 - iqr * w)
            # print(temp)

            o = origin_data[temp]
            n = origin_data[~temp]

            if len(o) >= g:
                print("number of outlier: ", len(o))
                # print(w)
                # print(o ,'\n\n', n)
                break
            w = w - 0.01

        return o, n  #

    def opt_jk(self, lim_j, lim_k):  #
        if self.OD == "default":
            o, n = self.outlierdetection(self.x, 2)
        elif self.OD == "KMOR":
            o, n = kmor(self.x,3,2,0.05)
        else:
            o, n = self.outlierdetection(self.x, 2)

        label_o, label_n = np.zeros(len(o)), np.ones(len(n))
        # print(o , n)
        # print(label_o, label_n)
        data = np.concatenate((o, n), axis=0)
        label = np.concatenate((label_o, label_n), axis=0)
        # print(data , label)

        gmean = np.zeros((lim_j, lim_k))

        kf = KFold(n_splits=2, random_state=random.randint(1, 100000), shuffle=True) #G-fold
        for train_index, test_index in kf.split(data):
            # print("TRAIN:", train_index, "TEST:", test_index)
            trainX, testX = data[train_index], data[test_index]
            trainy, testy = label[train_index], label[test_index]

            for pj in range(lim_j):
                for pk in range(lim_k):

                    op_classfiyer = JKNN(pj + 1, pk + 1)

                    # print(trainX , trainy)
                    # print(testX, testy)
                    op_classfiyer.fit(trainX)
                    pred_y = op_classfiyer.predict(testX)
                    # print("pred: ",pred_y)
                    # print("true label: ",testy)
                    tn, fp, fn, tp = confusion_matrix(testy, pred_y, labels=[0, 1]).ravel()
                    # print(tn, fp, fn, tp)
                    if (tp + fn) == 0:
                        tpr = 0
                    else:
                        tpr = tp / (tp + fn)
                    if (tn + fp) == 0:
                        tnr = 0
                    else:
                        tnr = tn / (tn + fp)
                    gmean[pj, pk] += (tpr * tnr) ** 0.5

        gmean = gmean / 2
        # print(gmean)
        best_index_of_jk = np.where(gmean == np.amax(gmean))
        # print(best_index_of_jk)
        # print(best_index_of_jk[0][0])
        # print(best_index_of_jk[1][0])
        self.set((best_index_of_jk[0][0]) + 1, (best_index_of_jk[1][0]) + 1)
        return (best_index_of_jk[0][0]) + 1, (best_index_of_jk[1][0]) + 1

    def opt_theta(self):  #

        if self.OD == "default":
            o, n = self.outlierdetection(self.x, 2)
        elif self.OD == "KMOR":
            o, n = kmor(self.x, 3, 2.5, 0.2)
        else:
            o, n = self.outlierdetection(self.x, 2)
        label_o, label_n = np.zeros(len(o)), np.ones(len(n))
        # print(o , n)
        # print(label_o, label_n)
        data = np.concatenate((o, n), axis=0)
        label = np.concatenate((label_o, label_n), axis=0)
        # print(data , label)
        theta_table = np.zeros(len(data))

        kf = KFold(n_splits=2, random_state=random.randint(1, 100000), shuffle=True) #G-fold
        counter = 0

        for train_index, test_index in kf.split(data):
            # print("TRAIN:", train_index, "TEST:", test_index)
            trainX, testX = data[train_index], data[test_index]
            trainy, testy = label[train_index], label[test_index]
            # print(len(testX),len(trainX) ,"??" , self.k + 1 )

            op_classfiyer = JKNN(self.j, self.k)  # self.j, self.k
            op_classfiyer.fit(trainX)
            neigh_dist, neigh_index = op_classfiyer.nn.kneighbors(testX, n_neighbors=self.j)  # self.j

            dj = np.sum(neigh_dist, axis=1)  # vector dj
            dj = dj / self.j  # self.j


            dk = np.array([0.0 for i in range(len(testX))])

            for i in range(len(neigh_index)):
                k_neigh_dist, k_neigh_index = op_classfiyer.nn.kneighbors(trainX[neigh_index[i]],
                                                                          n_neighbors=self.k + 1)  # self.k + 1
                dk[i] = np.sum(k_neigh_dist)

            dk = dk / (self.j * self.k)  # (self.j * self.k)


            result = (dj / dk)
            for i in range(len(result)):
                theta_table[test_index[i]] = result[i]

        print("theta_table completed")

        gmean = np.zeros(len(theta_table))

        for i in range(len(theta_table)):
            pred = np.where(theta_table > theta_table[i],0,1)

            tn, fp, fn, tp = confusion_matrix(label, pred, labels=[0, 1]).ravel()
            #print(tn, fp, fn, tp)

            if (tp + fn) == 0:
                tpr = 0
            else:
                tpr = tp / (tp + fn)
            if (tn + fp) == 0:
                tnr = 0
            else:
                tnr = tn / (tn + fp)
            gmean[i] += (tpr * tnr) ** 0.5

        best_index_of_theta = np.where(gmean == np.amax(gmean))
        # print(best_index_of_theta[0][0])
        self.set(self.j, self.k, theta_table[best_index_of_theta[0][0]])
        return theta_table[best_index_of_theta[0][0]]


    def fit(self, data):  #
        self.x = data
        self.nn.fit(self.x)

        return self


    def predict(self, test):  # 

        neigh_dist, neigh_index = self.nn.kneighbors(test, n_neighbors=self.j)

        dj = np.sum(neigh_dist, axis=1)  # vector dj
        dj = dj / self.j
        # print(dj)
        dk = np.array([0.0 for i in range(len(test))])
        count = 0
        for i in neigh_index:
            k_neigh_dist, k_neigh_index = self.nn.kneighbors(self.x[i], n_neighbors=self.k + 1)
            dk[count] = np.sum(k_neigh_dist)
            count += 1

        dk = dk / (self.j * self.k)
        # print(dk)

        # dk = np.where(dk == 0, 1, dk)
        result = ((dj / dk) < self.theta).astype(int)
        return result


    def show_paremeter(self):
        print("j = ", self.j, " k = ", self.k, " theta = ", self.theta)


def random_projection(xi, yi):
    i, j = xi.shape
    rp_matrix = np.zeros((j, i))
    base = 3 ** 0.5
    # print(base)
    for ii in range(j):
        for jj in range(i):
            v = random.randint(1, 6)
            if v == 1:  # 1/6
                rp_matrix[ii, jj] = base * 1
            elif v == 2:  # 1/6
                rp_matrix[ii, jj] = base * -1
            else:
                rp_matrix[ii, jj] = 0

    return rp_matrix.dot(xi), rp_matrix.dot(yi)



frame_data = pd.read_csv(sys.argv[1], low_memory=False) #----------------------dataset
print("number of original data = ", len(frame_data))
# print(frame_data)
# print(frame_data.isnull().sum())

frame_data = frame_data.dropna()
print("number of data = ", len(frame_data))

outcome_name = "outcome"

rank_list = frame_data.columns.tolist()
rank_list.remove("outcome")
print(rank_list)
X = frame_data[rank_list].values

y = frame_data[outcome_name].values
print(" Negative: ", np.count_nonzero(y == 0), " Positive: ", np.count_nonzero(y == 1))

pos_target = frame_data[frame_data[outcome_name] == 1]
neg_target = frame_data[frame_data[outcome_name] == 0]

pos_target = pos_target[rank_list].values
neg_target = neg_target[rank_list].values

#for i in range(1,11,1):
#
#  g = kmor(pos_target,i,2.5,0.1)
#  print("k = ",i," , outliers : " , np.count_nonzero(g == i))
scaler = StandardScaler()
pos_target = scaler.fit_transform(pos_target)
neg_target = scaler.fit_transform(neg_target)

neg_target = np.unique(neg_target, axis=0)
pos_target = np.unique(pos_target, axis=0)

av_auc=[]
av_acc = []
av_pre = []
av_rec = []
av_spe = []
av_gmean = []






for av in range(10):
    k = 5
    kf = KFold(n_splits=k, random_state=av*(random.randint(1,1000000)), shuffle=True)

    for train_index, test_index in kf.split(neg_target):
        # print("TRAIN:", train_index, "TEST:", test_index)
        trainX, testX = neg_target[train_index], neg_target[test_index]
        testy = np.ones(len(testX))
        label_p = np.zeros(len(pos_target))

        testX = np.concatenate((testX, pos_target), axis=0)
        testy = np.concatenate((testy, label_p), axis=0)

        ocnn = JKNN(1, 1, 1 )

        ocnn.fit(trainX)

        #tStart = time.time()
        #ocnn.opt_jk(5, 5 )
        #tEnd = time.time()
        #print((tEnd - tStart), "in opt-jk phase")

        # tStart = time.time()
        ocnn.opt_theta()
        # tEnd = time.time()
        # print((tEnd - tStart), "in opt-theta phase")

        tStart = time.time()
        pred = ocnn.predict(testX)
        tEnd = time.time()
        print((tEnd - tStart), "in predict phase")

        pred = np.where(pred == 0, 1, 0)  # reverse
        testy = np.where(testy == 0, 1, 0)  # reverse
        # print('Accuracy: {}'.format(accuracy_score(testy, pred)))
        av_acc.append(accuracy_score(testy, pred))
        # print("recall_score", metrics.recall_score(testy, pred))
        av_rec.append(metrics.recall_score(testy, pred))
        # print("precision score: ", precision_score(testy, pred, average='binary'))
        av_pre.append(precision_score(testy, pred, average='binary'))
        tn, fp, fn, tp = confusion_matrix(testy, pred).ravel()
        specificity = tn / (tn + fp)
        # print("specificity: ", specificity)
        av_spe.append(specificity)
        av_gmean.append((specificity * metrics.recall_score(testy, pred)) ** 0.5)


        av_auc.append(roc_auc_score(testy, pred))

print("auc :", sum(av_auc) / len(av_auc))

print("acc :", sum(av_acc) / len(av_acc))

print("pre :", sum(av_pre) / len(av_pre))

print("gmean :", sum(av_gmean) / len(av_gmean))

print("rec :", sum(av_rec) / len(av_rec))

print("spe :", sum(av_spe) / len(av_spe))

fpw = open("JKNN_" + sys.argv[1]+"_result.txt", "w")
fpw.write("auc = " + str(np.mean(av_auc)) + "\n")
fpw.write(str(np.quantile(av_auc, 0.025)) + " ~ " + str(np.quantile(av_auc, 0.975)) + " std = " + str(
        np.std(av_auc)) + " \n")
fpw.write("acc = " + str(np.mean(av_acc)) + "\n")
fpw.write(str(np.quantile(av_acc, 0.025)) + " ~ " + str(np.quantile(av_acc, 0.975)) + " std = " + str(
        np.std(av_acc)) + " \n")
fpw.write("pre = " + str(np.mean(av_pre)) + "\n")
fpw.write(str(np.quantile(av_pre, 0.025)) + " ~ " + str(np.quantile(av_pre, 0.975)) + " std = " + str(
        np.std(av_pre)) + " \n")
fpw.write("rec = " + str(np.mean(av_rec)) + "\n")
fpw.write(str(np.quantile(av_rec, 0.025)) + " ~ " + str(np.quantile(av_rec, 0.975)) + " std = " + str(
        np.std(av_rec)) + " \n")
fpw.write("spe = " + str(np.mean(av_spe)) + "\n")
fpw.write(str(np.quantile(av_spe, 0.025)) + " ~ " + str(np.quantile(av_spe, 0.975)) + " std = " + str(
        np.std(av_spe)) + " \n")
fpw.write("gmean = " + str(np.mean(av_gmean)) + "\n")
fpw.write(str(np.quantile(av_gmean, 0.025)) + " ~ " + str(np.quantile(av_gmean, 0.975)) + " std = " + str(
        np.std(av_gmean)) + " \n")

fpw.write("\n")
fpw.close()