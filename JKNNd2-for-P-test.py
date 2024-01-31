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
    dd = None
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
    temp = (U == k)


    # return U
    return U


#
class our_method():
    def __init__(self, theta=0.8, k=2, n0=0.1, r=2.5):    # 0 < theta < 1 , k => int
        self.theta = theta
        self.k = k
        self.n0 = n0
        self.r = r
        self.U = None
        self.x = None
        self.group = None    # store each k group data
        self.near_group = None

    def fit(self, x):
        self.x = x

        while(1):
            self.group = []
            self.near_group = []
            check = True

            self.U = kmor(self.x, self.k, self.r, self.n0)
            for i in range(self.k):
                index = np.arange(len(self.x))[self.U == i]
                if len(index) < 2:
                    check = False
                    break
                self.group.append(self.x[index])
                self.near_group.append(NearestNeighbors(n_neighbors=1).fit(self.x[index]))

            self.group = np.array(self.group)
            if check:
                break
            else:
                self.k -= 1
                continue




    def predict(self,test):

        result = np.zeros(len(test),dtype=int)
        count = 0
        for item in test:    #each data
            for i in range(self.k):#search each group
                k_neigh_dist, k_neigh_index = self.near_group[i].kneighbors([item], n_neighbors=1)

                # print(k_neigh_dist[0][0],k_neigh_index[0][0])
                distance_table = np.linalg.norm(self.group[i][k_neigh_index[0][0]] - self.group[i], axis=1)
                #print("theta= ", np.quantile(distance_table,self.theta))
                if k_neigh_dist[0][0] < np.quantile(distance_table,self.theta):
                    result[count] = 1
                    break






            count += 1
        return result



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


scaler = StandardScaler()
pos_target = scaler.fit_transform(pos_target)
neg_target = scaler.fit_transform(neg_target)

neg_target = np.unique(neg_target, axis=0)
pos_target = np.unique(pos_target, axis=0)





p_auc = np.zeros((5,20))
p_acc = np.zeros((5,20))
p_rec = np.zeros((5,20))
p_spe = np.zeros((5,20))
p_gmean = np.zeros((5,20))




for test_j in range(1,6):
    for test_k in range(0,20):
      av_auc=[]
      av_acc = []
      av_pre = []
      av_rec = []
      av_spe = []
      av_gmean = []
      for av in range(2):
        k = 5
        kf = KFold(n_splits=k, random_state=av*(random.randint(1,1000000)), shuffle=True)

        for train_index, test_index in kf.split(pos_target):
                # print("TRAIN:", train_index, "TEST:", test_index)
          trainX, testX = pos_target[train_index], pos_target[test_index]
          testy = np.ones(len(testX))
          label_p = np.zeros(len(neg_target))

          testX = np.concatenate((testX, neg_target), axis=0)
          testy = np.concatenate((testy, label_p), axis=0)

          ocnn = our_method(0.01+0.02*test_k, test_j, 0.03, 2.5)

          ocnn.fit(trainX)

                # tStart = time.time()
                # ocnn.opt_jk(5, 5 )
                # tEnd = time.time()
                # print((tEnd - tStart), "in opt-jk phase")

                # tStart = time.time()
                # ocnn.opt_theta()
                # tEnd = time.time()
                # print((tEnd - tStart), "in opt-theta phase")

          tStart = time.time()
          pred = ocnn.predict(testX)
          tEnd = time.time()
          print((tEnd - tStart), "in predict phase")

                # pred = np.where(pred == 0, 1, 0)  # reverse
                # testy = np.where(testy == 0, 1, 0)  # reverse
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
          
        print("K = ",test_j, " Theta = ", (0.01+0.02*test_k) )
        print("auc :", sum(av_auc) / len(av_auc))
        p_auc[test_j-1,test_k] = sum(av_auc) / len(av_auc)
        print("acc :", sum(av_acc) / len(av_acc))
        p_acc[test_j-1, test_k] = sum(av_acc) / len(av_acc)
        print("pre :", sum(av_pre) / len(av_pre))

        print("gmean :", sum(av_gmean) / len(av_gmean))
        p_gmean[test_j-1, test_k] = sum(av_gmean) / len(av_gmean)
        print("rec :", sum(av_rec) / len(av_rec))
        p_rec[test_j-1, test_k] = sum(av_rec) / len(av_rec)
        print("spe :", sum(av_spe) / len(av_spe))
        p_spe[test_j-1, test_k] = sum(av_spe) / len(av_spe)


np.save('p_auc', p_auc)
np.save('p_acc', p_acc)
np.save('p_rec', p_rec)
np.save('p_spe', p_spe)
np.save('p_gmean', p_gmean)



print("auc = ")
print(p_auc)

print("acc = ")
print(p_acc)

print("rec = ")
print(p_rec)

print("spe = ")
print(p_spe)

print("gmean = ")
print(p_gmean)