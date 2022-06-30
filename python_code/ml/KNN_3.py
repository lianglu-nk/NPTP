import multiprocessing as mp
import os
import warnings
from random import shuffle
from rdkit.Chem import MACCSkeys
import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles,AllChem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit import DataStructs
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix, silhouette_score, average_precision_score
import datetime
import argparse
import pandas as pd
import itertools
import numpy as np
import pickle
warnings.filterwarnings("ignore")
#Hyperparametric dictionary
dictionary0 = {
    'n_neighbors':[1,3,5,7,9,11]
}
#Root directory
basePath=os.getcwd()
#Hyperparameters combination
hyperParams0 = pd.DataFrame(list(itertools.product(*dictionary0.values())), columns=dictionary0.keys())
hyperParams0.index=np.arange(len(hyperParams0.index.values))
#External parameters input
parser = argparse.ArgumentParser()
parser.add_argument("-cl_file", help="cl per target", type=str, default=basePath+'/noweakchembl26end/ML/cl/')
parser.add_argument("-pertarget_file", help="smi per target", type=str, default=basePath+'/noweakchembl26end/ML/pertargetdata/')
parser.add_argument("-datasetNames", help="Dataset Name",type=str, default="ecfp6fcfp6MACCS")
parser.add_argument("-saveBasePath", help="saveBasePath", type=str, default=basePath+'/noweakchembl26end/ML/res_noweakchembl26end/')
parser.add_argument("-ofolds", help="Outer Folds", nargs='+', type=int, default=[0, 1, 2, 3, 4])
parser.add_argument("-ifolds", help="Inner Folds", nargs='+', type=int, default=[0, 1, 2, 3, 4])
parser.add_argument("-pStart", help="Parameter Start Index", type=int, default=0)
parser.add_argument("-pEnd", help="Parameter End Index", type=int, default=6)
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

#The optional parameter is selected according to the AUC value
def bestSettingsSimple(perfFiles, nrParams):
    aucFold=[]
    for outind in range(0,5):
        for foldInd in range(0, 4):
            aucParam=[]
            for paramNr in range(0, nrParams):
                #try:
                saveFile=open(perfFiles[outind][foldInd][paramNr], "rb")
                aucRun=pickle.load(saveFile)
                saveFile.close()
                #except:
                #  pass
                if(len(aucRun)>0):
                    aucParam.append(aucRun[-1])
        
            aucParam=np.array(aucParam)
        
            if(len(aucParam)>0):
                aucFold.append(aucParam)
    aucFold=np.array(aucFold)
    aucMean=np.nanmean(aucFold, axis=0)
    paramInd=np.nanmean(aucMean, axis=1).argmax()
  
    return (paramInd, aucMean, aucFold)
#Extraction of cluster information, characteristic information, activity information and target name corresponding to each target 
def ClusterCV(csv_file):
    tar_id = csv_file.split('.')[0]
    file_name = pertarget_file + csv_file
    clusterSampleFilename = os.path.join(cl_file, 'cl' + tar_id + ".info")
    chembl_data = file_reader(file_name)
    target_name = chembl_data.iloc[0,0]
    labels = chembl_data.active_label
    features = batchECFP(chembl_data.canonical_smiles)
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
        fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        fp_1 = GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=True)
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
    # method one: file_list = os.listdir(file_folder)
    for root, dirs, file_list in os.walk(file_folder):
        return file_list

#CSV reading 
def file_reader(file_path):
    data = pd.read_csv(file_path)
    return data

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
    dbgPath=save_path+"dbg/"
    if not os.path.exists(dbgPath):
        os.makedirs(dbgPath)
    # modeling

    
    savePrefix0=target_name+'_model'
    savePrefix=save_path+savePrefix0
    dbgOutput=open(dbgPath+savePrefix0+".dbg", "w")
    perfFiles=[]
    #All compounds are modeled with selected parameters
    for outerFold in compOuterFolds:
        perfFiles.append([])
        for innerFold in compInnerFolds:
            if innerFold == outerFold:
                continue
            pperf = []
            for paramNr in range(0, hyperParams0.shape[0]):
                pperf.append(save_path+"o"+'{0:04d}'.format(outerFold+1)+"_i"+'{0:04d}'.format(innerFold+1)+"_p"+'{0:04d}'.format(hyperParams0.index.values[paramNr])+".test.auc.pckl")
            perfFiles[-1].append(pperf)
    paramNr, perfTable, perfTableOrig=bestSettingsSimple(perfFiles, hyperParams0.shape[0])
    print(hyperParams0.iloc[paramNr], file=dbgOutput)
    print(perfTable, file=dbgOutput)
    dbgOutput.close()        

    n_neighbors1 = hyperParams0.iloc[paramNr].n_neighbors
    knn = KNeighborsClassifier(n_neighbors=n_neighbors1)

    # Get training data
    train_index = np.where(folds != 6)[0]
    #print(train_index)
    # Get train samples
    X_train = features[train_index, :]
    y_train = active_label.iloc[train_index]
    # Modeling 
    knn.fit(X_train, y_train)
    
    saveFilename = savePrefix + "_knn.pckl"
    joblib.dump(knn,saveFilename)

startime = datetime.datetime.now()
print("Start time", startime)
#Active data file for each target
folder_path = pertarget_file
files_list = get_file_list(folder_path)
#Number of targets corresponds to number of tasks
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
