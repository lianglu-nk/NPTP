import datetime
import pandas as pd
import numpy as np
import argparse
import os
from numpy import mean
from numpy import std
from rdkit.Chem import MACCSkeys
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import multiprocessing as mp
from rdkit.Chem import MolFromSmiles,AllChem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

#Root directory
basePath=os.getcwd()

#External parameters input
parser = argparse.ArgumentParser()
parser.add_argument("-cl_file", help="cl per target", type=str, default=basePath+'/noweakchembl29end/ML/cl/')
parser.add_argument("-pertarget_file", help="smi per target", type=str, default=basePath+'/noweakchembl29end/ML/pertargetdata/')
parser.add_argument("-saveBasePath", help="saveBasePath", type=str, default=basePath+'/noweakchembl29end/ML/res_noweakchembl29end_random/')
args = parser.parse_args()
pertarget_file = args.pertarget_file
saveBasePath = args.saveBasePath
cl_file = args.cl_file

#CSV reading
def file_reader(file_path):
    data = pd.read_csv(file_path)
    return data

#Read file information in a folder
def get_file_list(file_folder):
    # method one: file_list = os.listdir(file_folder)
    for root, dirs, file_list in os.walk(file_folder):
        return file_list

#SMILES to fingerprint feature conversion / Different combinations of fingerprint features
def batchECFP(smiles, radius=3, nBits=2048):
    smiles = np.array(smiles)
    n = len(smiles)
    fingerprints_0 = np.zeros((n, nBits), dtype=int)
    fingerprints_1 = np.zeros((n, nBits), dtype=int)
    MACCSArray = []
    for i in range(n):
        mol = MolFromSmiles(smiles[i])
        fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        fp_1 = GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=True)
        MACCSArray.append(MACCSkeys.GenMACCSKeys(mol))
        fingerprints_0[i] = np.array(list(fp.ToBitString()))
        fingerprints_1[i] = np.array(list(fp_1.ToBitString()))
    fingerprints_2 = np.array(MACCSArray)
    fingerprints = np.hstack((fingerprints_0, fingerprints_1, fingerprints_2))
    return fingerprints

def data_split_modeling(target_file):
    file_name = pertarget_file + target_file
    chembl_data = file_reader(file_name)
    target_name = chembl_data.iloc[0, 0]
    X = batchECFP(chembl_data.canonical_smiles)
    y = chembl_data.active_label
    # enumerate splits
    cv_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=1)
    outer_results = list()
    save_path = saveBasePath + 'ML/RF/' + target_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dbgPath1out = open(save_path+"single.dbg","w")
    for train_ix, test_ix in cv_outer.split(X,y):
        # split data
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        # configure the cross-validation procedure
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
        cv_inner = skf.split(X_train, y_train)
        # define the model
        model = RandomForestClassifier(random_state=1)
        # define search space
        space = {
            'n_estimators': [10, 50, 100, 300, 700, 1000],
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [1, 3, 5, 7, 9]
        }
        # define search
        search = GridSearchCV(model, space, scoring='roc_auc', cv=cv_inner,refit=True)
        # execute search
        result = search.fit(X_train, y_train)
        # get the best performing model fit on the whole training setc
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict_proba(X_test)
        # evaluate the model
        auc = roc_auc_score(y_test, yhat[:, 1])
        # store the result
        outer_results.append(auc)
        # report progress
        print('%f,%f,%s' % (auc, result.best_score_, result.best_params_), file=dbgPath1out)
    # summarize the estimated performance of the model
    dbgPath2out = open(save_path+"mean.dbg", "w")
    print('%f,%f' % (mean(outer_results), std(outer_results)), file=dbgPath2out)
    dbgPath1out.close()
    dbgPath2out.close()

startime = datetime.datetime.now()
print("Start time", startime)
folder_path = pertarget_file
files_list = sorted(get_file_list(folder_path))
# data_split_modeling(files_list[0])
p = mp.Pool(processes=949)
for tar_file in files_list:
    result = p.apply_async(data_split_modeling, args=(tar_file,))  # distribute one task to one pool
p.close()  # finished load task
p.join()  # start
print("Sub-process(es) done.")
endtime = datetime.datetime.now()
print("End time", endtime)
costime = endtime - startime
print("Cost time", costime)

