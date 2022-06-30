import pandas as pd
import itertools
import numpy as np
import pickle
import os
import multiprocessing as mp
import os
import warnings
from random import shuffle
from rdkit.Chem import MACCSkeys
import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import joblib
import argparse

basePath=os.getcwd()
#External parameters input
parser = argparse.ArgumentParser()
parser.add_argument("-smiles_file", help="smiles needed to be predicted", type=str, default=basePath+'/data/test.csv')
parser.add_argument("-method_name", help="method name", type=str, default='KNN')
parser.add_argument("-datatset_type", help="dataset type", type=str, default='chembl29')
parser.add_argument("-saveBasePath", help="saveBasePath", type=str, default=basePath+'/result/')
args = parser.parse_args()
smiles_file = args.smiles_file
method_name = args.method_name
datatset_type = args.datatset_type
saveBasePath = args.saveBasePath
method_path = basePath+'/model/'+datatset_type+'/'+method_name+'/'

#SMILES to fingerprint feature conversion
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
    return fingerprints


    
filepath = method_path
files_list = os.listdir(filepath)
num = len(files_list)

pred_data = pd.read_csv(smiles_file)
mol_features = batchECFP(pred_data.smiles)


for i in range(1):
  target= filepath + files_list[i]
  tar_id= files_list[i].split('_')[0]
  #Load model 
  tar_model = joblib.load(target)
  y_pred = tar_model.predict(mol_features)
  y_pred_proba = tar_model.predict_proba(mol_features)
  dataout = pd.DataFrame(columns=[tar_id], data=y_pred)

  
for i in range(1,num):
  target= filepath + files_list[i]
  tar_id= files_list[i].split('_')[0]
  #Load model 
  tar_model = joblib.load(target)
  y_pred = tar_model.predict(mol_features)
  y_pred_proba = tar_model.predict_proba(mol_features)
  dataout[tar_id] = y_pred

dataout.to_csv(saveBasePath+datatset_type+method_name+'.csv', index=True, header=True)
