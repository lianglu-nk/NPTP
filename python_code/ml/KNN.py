import multiprocessing as mp
import os
import warnings
import datetime
import argparse
import pandas as pd
import itertools
import numpy as np
import pickle

from rdkit.Chem import MACCSkeys
from rdkit.Chem import MolFromSmiles,AllChem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
warnings.filterwarnings("ignore")

dictionary0 = {
    "n_neighbors":range(1,21,2)
}
#Root directory
basePath=os.getcwd()
#Hyperparameters combination
hyperParams0 = pd.DataFrame(list(itertools.product(*dictionary0.values())), columns=dictionary0.keys())
hyperParams0.index=np.arange(len(hyperParams0.index.values))
#External parameters input
parser = argparse.ArgumentParser()
parser.add_argument("-datasetNames", help="Dataset Name",type=str, default="ecfp6fcfp6MACCS")
parser.add_argument("-cl_file", help="cl per target", type=str, default=basePath+"/noweakchembl29end/ML/cl/")
parser.add_argument("-pertarget_file", help="smi per target", type=str, default=basePath+"/noweakchembl29end/ML/pertargetdata/")
parser.add_argument("-saveBasePath", help="saveBasePath", type=str, default=basePath+"/noweakchembl29end/ML/res_noweakchembl29end/")
parser.add_argument("-ofolds", help="Outer Folds", nargs='+', type=int, default=[0, 1, 2, 3, 4])
parser.add_argument("-ifolds", help="Inner Folds", nargs='+', type=int, default=[0, 1, 2, 3, 4])
parser.add_argument("-pStart", help="Parameter Start Index", type=int, default=0)
parser.add_argument("-pEnd", help="Parameter End Index", type=int, default=10)
args = parser.parse_args()
cl_file = args.cl_file
pertarget_file = args.pertarget_file
datasetNames = args.datasetNames
saveBasePath = args.saveBasePath
compOuterFolds = args.ofolds
compInnerFolds = args.ifolds
paramStart = args.pStart
paramEnd = args.pEnd

#Parameter index
compParams = list(range(paramStart, paramEnd))

#Extraction of cluster information, characteristic information, activity information and target name corresponding to each target 
def ClusterCV(csv_file):
    #Read active files in sequence
    tar_id = csv_file.split(".")[0]
    file_name = pertarget_file + csv_file
    #The clustering file corresponding to the active file
    clusterSampleFilename = os.path.join(cl_file, "cl" + tar_id + ".info")
    #Extract information from active files
    chembl_data = file_reader(file_name)
    target_name = chembl_data.iloc[0,0]
    labels = chembl_data.active_label
    features = batchECFP(chembl_data.canonical_smiles)
    #Clustering information extraction
    clusterTab = pd.read_csv(clusterSampleFilename, header=None, index_col=False, sep=",")
    df = clusterTab.values
    folds = df[:, 0]
    return folds, features, labels, target_name

#SMILES to fingerprint feature conversion / Different combinations of fingerprint features
def batchECFP(smiles, radius=3, nBits=2048):
    smiles = np.array(smiles)
    n = len(smiles)
    fingerprints_0 = np.zeros((n, nBits), dtype=int)
    fingerprints_1 = np.zeros((n, nBits), dtype=int)
    MACCSArray = []
    for i in range(n):
        mol = MolFromSmiles(smiles[i])
        #ecfp6
        fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        #fcfp6
        fp_1 = GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=True)
        #MACCS
        MACCSArray.append(MACCSkeys.GenMACCSKeys(mol))
        fingerprints_0[i] = np.array(list(fp.ToBitString()))
        fingerprints_1[i] = np.array(list(fp_1.ToBitString()))
    fingerprints_2 = np.array(MACCSArray)
    fingerprints = np.hstack((fingerprints_0,fingerprints_1,fingerprints_2))
    fingerprints_3=np.hstack((fingerprints_0,fingerprints_1))
    fingerprints_4=np.hstack((fingerprints_0,fingerprints_2))
    fingerprints_5=np.hstack((fingerprints_1,fingerprints_2))
    if datasetNames=="ecfp6fcfp6MACCS":
        fingerprints_out=fingerprints
    elif datasetNames=="ecfp6":
        fingerprints_out=fingerprints_0
    elif datasetNames=="fcfp6":
        fingerprints_out=fingerprints_1
    elif datasetNames=="MACCS":
        fingerprints_out=fingerprints_2
    elif datasetNames=="ecfp6fcfp6":
        fingerprints_out=fingerprints_3
    elif datasetNames=="ecfp6MACCS":
        fingerprints_out=fingerprints_4
    elif datasetNames=="fcfp6MACCS":
        fingerprints_out=fingerprints_5
    
    return fingerprints_out

#Read file information in a folder
def get_file_list(file_folder):
    for root, dirs, file_list in os.walk(file_folder):
        return file_list

#CSV reading 
def file_reader(file_path):
    data = pd.read_csv(file_path)
    return data

#Clustering nested cross validation
def data_split_modeling(target_file):
    target_id = target_file.split('.')[0]
    cluster_res = ClusterCV(csv_file=target_file)
    folds = cluster_res[0]
    features = cluster_res[1]
    active_label = cluster_res[2]
    target_name = cluster_res[3]
    save_path = saveBasePath+"/ML/"+datasetNames+"/KNN/" + target_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #Different parameter combinations for nested cross validation
    for paramNr in compParams:
        for outerFold in compOuterFolds:
            for innerFold in compInnerFolds:
                if innerFold == outerFold:
                    continue
                try:
                    roc_auc = []
                    #The file is named by outer fold, inner fold, and parameter index
                    savePrefix0 = "o" + '{0:04d}'.format(outerFold + 1) + "_i" + '{0:04d}'.format(
                        innerFold + 1) + "_p" + '{0:04d}'.format(hyperParams0.index.values[paramNr])
                    savePrefix = save_path + savePrefix0
                    #determine parameter / modeling
                    n_neighbors1 = hyperParams0.iloc[paramNr].n_neighbors
                    knn = KNeighborsClassifier(n_neighbors=n_neighbors1)

                    # Get training and testing index
                    test_index = np.where(folds == innerFold)[0]
                    train_index = np.where((folds != outerFold) & (folds != innerFold))[0]
                    # Get train and test samples
                    X_test = features[test_index, :]
                    X_train = features[train_index, :]
                    y_test = active_label.iloc[test_index]
                    y_train = active_label.iloc[train_index]
                    # Modeling and Evaluate
                    knn.fit(X_train, y_train)
                    y_pred = knn.predict(X_test)
                    # compute roc-auc
                    y_pred_proba = knn.predict_proba(X_test)
                    roc_auc.append(roc_auc_score(y_test, y_pred_proba[:, 1]))
                    a = np.array(roc_auc)
                    reportTestAUC = []
                    reportTestAUC.append(a)
                    saveFilename = savePrefix + ".test.auc.pckl"
                    saveFile = open(saveFilename, "wb")
                    pickle.dump(reportTestAUC, saveFile)
                    saveFile.close()
                except:
                    with open(saveBasePath+"/ML/"+datasetNames+"/KNN/" +target_name+"failed_target", "a+") as failed:
                        failed.write(target_name + " " + "failed" + "\n")

startime = datetime.datetime.now()
print("Start time", startime)
#Active data file for each target
folder_path = pertarget_file
files_list = get_file_list(folder_path)
#Number of targets corresponds to number of tasks
p = mp.Pool(processes=949)
for tar_file in files_list:
    # data_split_modeling(tar_file)
    result = p.apply_async(data_split_modeling, args=(tar_file,))  # distribute one task to one pool
p.close()  # finished load task
p.join()  # start
print("Sub-process(es) done.")
endtime = datetime.datetime.now()
print("End time", endtime)
costime = endtime - startime
print("Cost time", costime)
