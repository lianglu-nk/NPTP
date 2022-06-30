#implementation similar to http://www.bioinf.jku.at/research/lsc/
#with the following copyright information:
#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

import math
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.sparse
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
import argparse
import itertools

file_path_now=os.getcwd()
catalog=file_path_now.split('python_code')[0]

np.set_printoptions(threshold=1000)
np.set_printoptions(linewidth=160)
np.set_printoptions(precision=4)
np.set_printoptions(edgeitems=15)
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("-destPath", help="path for out data", type=str,default=catalog+ "/test_data/")
parser.add_argument("-infilna1", help="input file1 name", type=str,default="chembl29")
parser.add_argument("-infilna2", help="input file2 name", type=str,default="train_test")
#parser.add_argument("-tarnum", help="target number", type=str,default=3)

args = parser.parse_args()

inputPathname1 = args.infilna1
targetSampleInfo = args.infilna2
dataPathSave = args.destPath

#tar_num = args.tarnum

projectPathname = catalog+'/'+inputPathname1+'/out_put_file'

if not os.path.exists(dataPathSave):
    os.makedirs(dataPathSave)

target_file=dataPathSave+'target_number.csv'
target_file_read = pd.read_csv(target_file)
tar_num=target_file_read.iloc[0,0]
print(tar_num)

clusterInfo = "cl1";


chemPathname = os.path.join(projectPathname, "chemFeatures");
clusterPathname = os.path.join(chemPathname, "cl");
schemPathname = os.path.join(chemPathname, "s");
trainPathname = os.path.join(projectPathname, "train");
sampleIdFilename = os.path.join(chemPathname, "SampleIdTable.txt")

targetSampleFilename = os.path.join(trainPathname, targetSampleInfo + ".csv")
clusterSampleFilename = os.path.join(clusterPathname, clusterInfo + ".info")


sampleIdFilename = os.path.join(chemPathname, 'SampleIdTable.txt')
tpSampleIdToName = np.genfromtxt(sampleIdFilename, dtype=str)

clusterTab = pd.read_csv(clusterSampleFilename, header=None, index_col=False, sep=" ")
folds = [clusterTab.loc[clusterTab.iloc[:, 0] == 0].iloc[:, 1].values,
         clusterTab.loc[clusterTab.iloc[:, 0] == 1].iloc[:, 1].values,
         clusterTab.loc[clusterTab.iloc[:, 0] == 2].iloc[:, 1].values]
f = open(dataPathSave + 'folds0.pckl', "wb")
pickle.dump(folds, f, -1)
f.close()
lookup = clusterTab.set_index(1)

trainTab = pd.read_csv(targetSampleFilename, header=None, index_col=False, sep=",")
trainTab = trainTab.drop_duplicates()

classTypes = np.array([1, 3])
clusterNames = np.sort(np.unique(lookup))
targetNames = np.sort(np.unique(trainTab.iloc[:, 2].values))

classNumbers = []
for clusterName in clusterNames:
    classNumbersCluster = []
    for classType in classTypes:
        trainTabSel = trainTab.loc[(lookup.iloc[trainTab.iloc[:, 1].values] == clusterName).values[:, 0]]
        eq = trainTabSel.loc[trainTabSel.iloc[:, 0] == classType].iloc[:, 2].value_counts()
        ser = pd.Series(np.zeros(len(targetNames), dtype=np.int64))
        ser.index = targetNames
        ser[eq.index.values] = eq.values
        classNumbersCluster.append(ser)
    classNumbers.append(np.array(classNumbersCluster))
classNumbers = np.array(classNumbers)
selTargetNames = targetNames
trainTabSel = trainTab.loc[np.in1d(trainTab.iloc[:, 0].values, [1, 3])]
trainTabSel = trainTabSel.loc[np.in1d(trainTabSel.iloc[:, 2].values, selTargetNames)]

sampleAnnInd = pd.Series(data=np.arange(len(tpSampleIdToName)), index=np.arange(len(tpSampleIdToName)))
targetAnnInd = pd.Series(data=np.arange(len(selTargetNames)), index=selTargetNames)
annMat = scipy.sparse.coo_matrix(
    (trainTabSel.iloc[:, 0], (sampleAnnInd[trainTabSel.iloc[:, 1]], targetAnnInd[trainTabSel.iloc[:, 2]])),
    shape=(sampleAnnInd.max() + 1, targetAnnInd.max() + 1))
annMat.data[annMat.data < 2] = -1
annMat.data[annMat.data > 2] = 1
annMat.eliminate_zeros()
annMat = annMat.tocsr()
annMat.sort_indices()
f = open(dataPathSave + 'labelsHard.pckl', "wb")
pickle.dump(annMat, f, -1)
pickle.dump(sampleAnnInd, f, -1)
pickle.dump(targetAnnInd, f, -1)
f.close()
targetMat = annMat
targetMat = targetMat
targetMat = targetMat.copy().tocsr()
targetMat.sort_indices()
targetAnnInd = targetAnnInd
targetAnnInd = targetAnnInd - targetAnnInd.min()

#folds = [np.intersect1d(fold, sampleAnnInd.index.values).tolist() for fold in folds]
targetMatTransposed = targetMat[sampleAnnInd[list(itertools.chain(*folds))]].T.tocsr()
targetMatTransposed.sort_indices()
trainPosOverall = np.array([np.sum(targetMatTransposed[x].data > 0.5) for x in range(targetMatTransposed.shape[0])])
trainNegOverall = np.array([np.sum(targetMatTransposed[x].data < -0.5) for x in range(targetMatTransposed.shape[0])])
fold_1 = folds[0]
fold_2 = folds[1]
fold_3 = folds[2]
targetMatTransposed_1 = targetMat[sampleAnnInd[fold_1]].T.tocsr()
targetMatTransposed_2 = targetMat[sampleAnnInd[fold_2]].T.tocsr()
targetMatTransposed_3 = targetMat[sampleAnnInd[fold_3]].T.tocsr()
targetMatTransposed_1.sort_indices()
targetMatTransposed_2.sort_indices()
targetMatTransposed_3.sort_indices()
trainPos_1 = np.array([np.sum(targetMatTransposed_1[x].data > 0.5) for x in range(targetMatTransposed_1.shape[0])])
trainNeg_1 = np.array([np.sum(targetMatTransposed_1[x].data < -0.5) for x in range(targetMatTransposed_1.shape[0])])
trainPos_2 = np.array([np.sum(targetMatTransposed_2[x].data > 0.5) for x in range(targetMatTransposed_2.shape[0])])
trainNeg_2 = np.array([np.sum(targetMatTransposed_2[x].data < -0.5) for x in range(targetMatTransposed_2.shape[0])])
trainPos_3 = np.array([np.sum(targetMatTransposed_3[x].data > 0.5) for x in range(targetMatTransposed_3.shape[0])])
trainNeg_3 = np.array([np.sum(targetMatTransposed_3[x].data < -0.5) for x in range(targetMatTransposed_3.shape[0])])
for i in range(tar_num):
    if trainPos_1[i] >= 1:
        pass
    if trainPos_1[i] <1:
        print("false")
for i in range(tar_num):
    if trainPos_2[i] >= 1:
        pass
    if trainPos_2[i] <1:
        print("false")
for i in range(tar_num):
    if trainPos_3[i] >= 1:
        pass
    if trainPos_3[i] <1:
        print("false")
for i in range(tar_num):
    if trainNeg_1[i] >= 1:
        pass
    if trainNeg_1[i] <1:
        print("false")
for i in range(tar_num):
    if trainNeg_2[i] >= 1:
        pass
    if trainNeg_2[i] <1:
        print("false")
for i in range(tar_num):
    if trainNeg_3[i] >= 1:
        pass
    if trainNeg_3[i] <1:
        print("false")
print(trainNeg_1)
print(trainPos_1)
print(trainNeg_2)
print(trainPos_2)
print(trainNeg_3)
print(trainPos_3)
