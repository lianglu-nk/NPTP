import math
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.sparse
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pickle
import os
import argparse
import rdkit
from rdkit.Chem import MACCSkeys

file_path_now=os.getcwd()
catalog=file_path_now.split('python_code')[0]
print(catalog)
parser = argparse.ArgumentParser()
parser.add_argument("-supplfile", help="sdf file path", type=str,default=catalog+"/test_data/test.sdf")
parser.add_argument("-lstmfeature", help="lstm sparse feature", default=False)
parser.add_argument("-featureoutfile", help="pckl file path", type=str, default=catalog+"/test_data/")
parser.add_argument("-featureoutname", help="pckl file name", type=str, default="test")
args = parser.parse_args()

sdf_supply = args.supplfile
lstmfeature = args.lstmfeature
featureoutfile = args.featureoutfile
featureoutname = args.featureoutname
feature_format='dense'

suppl = Chem.SDMolSupplier(sdf_supply)
ms = [x for x in suppl if x is not None]
n = len(ms)
ecfpMat = np.zeros((n, 2048), dtype=int)
for i in range(n):
    fp = AllChem.GetMorganFingerprintAsBitVect(ms[i], 3, 2048)
    ecfpMat[i] = np.array(list(fp.ToBitString()))
if lstmfeature:
    ecfpMat = scipy.sparse.csr_matrix(ecfpMat)
    feature_format='sparse'
sampleECFPInd = pd.Series(data=np.arange(len(ms)), index=np.arange(len(ms)))
f = open(featureoutfile+featureoutname+'ecfp6'+feature_format+'.pckl', "wb")
pickle.dump(ecfpMat, f, -1)
pickle.dump(sampleECFPInd, f, -1)
f.close()


fcfpMat = np.zeros((n, 2048), dtype=int)
for i in range(n):
    fp = AllChem.GetMorganFingerprintAsBitVect(ms[i], 3, 2048, useFeatures=True)
    fcfpMat[i] = np.array(list(fp.ToBitString()))
if lstmfeature:
    fcfpMat = scipy.sparse.csr_matrix(fcfpMat)
sampleFCFPInd = pd.Series(data=np.arange(len(ms)), index=np.arange(len(ms)))
f = open(featureoutfile+featureoutname+'fcfp6'+feature_format+'.pckl', "wb")
pickle.dump(fcfpMat, f, -1)
pickle.dump(sampleFCFPInd, f, -1)
f.close()

if not lstmfeature:
    MACCSArray = []
    for x in ms:
        MACCSArray.append(MACCSkeys.GenMACCSKeys(x))
    MACCSMat=np.array(MACCSArray)
    sampleMACCSInd = pd.Series(data=np.arange(len(ms)), index=np.arange(len(ms)))
    f=open(featureoutfile+featureoutname+'MACCSFNN'+feature_format+'.pckl', "wb")
    pickle.dump(MACCSMat, f, -1)
    pickle.dump(sampleMACCSInd, f, -1)
    f.close()
