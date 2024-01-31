import numpy as np
import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import sys
from sklearn import metrics
import time
import math
from sklearn.svm import OneClassSVM

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

# for i in range(1,11,1):
#
#     g = kmor(pos_target,i,3,0.5)
#     print("k = ",i," , outliers : " , np.count_nonzero(g == i))
scaler = StandardScaler()
pos_target = scaler.fit_transform(pos_target)
neg_target = scaler.fit_transform(neg_target)

neg_target = np.unique(neg_target, axis=0)
pos_target = np.unique(pos_target, axis=0)


av_auc = []
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

        clf = OneClassSVM(gamma='auto').fit(trainX)


        pred = clf.predict(testX)
        pred = np.where(pred == -1, 0, pred)  # reverse

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

fpw = open("OC-SVM_" + sys.argv[1]+"_result.txt", "w")
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