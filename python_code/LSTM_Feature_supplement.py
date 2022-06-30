#Parts of code in this file have been taken (copied) from https://github.com/ml-jku/lsc
#Copyright (C) 2018 Andreas Mayr

import rdkit
import rdkit.Chem
import os
import numpy as np
import pickle
import pandas as pd
import scipy
import scipy.io
import scipy.sparse
import rdkit.Chem.MACCSkeys
import argparse
file_path_now=os.getcwd()
catalog=file_path_now.split('python_code')[0]

parser = argparse.ArgumentParser()
parser.add_argument("-supplfile", help="sdf file path", type=str, default=catalog+"/test_data/test.sdf")

parser.add_argument("-featureoutfile", help="pckl file path", type=str, default=catalog+"/test_data/")
parser.add_argument("-featureoutname", help="pckl file name", type=str, default="test")
args = parser.parse_args()

sdf_supply = args.supplfile

featureoutfile = args.featureoutfile
featureoutname = args.featureoutname

chemblMols=rdkit.Chem.SDMolSupplier(sdf_supply, True, False, False)
chemblMolsArr=[]
for ind in range(len(chemblMols)):
  mol=chemblMols[ind]
  chemblMolsArr.append(mol)

chemblMolsArrCopy=np.array(chemblMolsArr)

fileObject=open(featureoutfile+featureoutname+"LSTM.pckl",'wb')
pickle.dump(chemblMolsArrCopy, fileObject)
fileObject.close()

chemblSmilesArr=[]
for ind in range(len(chemblMolsArr)):
  if chemblMolsArr[ind] is not None:
    chemblSmilesArr.append(rdkit.Chem.MolToSmiles(chemblMolsArr[ind]))
  else:
    chemblSmilesArr.append(rdkit.Chem.MolToSmiles(rdkit.Chem.MolFromSmiles("")))

f=open(featureoutfile+featureoutname+'Smiles.pckl', "wb")
pickle.dump(chemblSmilesArr, f)
f.close()



chemblMACCSArr=[]
for ind in range(len(chemblMolsArr)):
  if chemblMolsArr[ind] is not None:
    chemblMACCSArr.append(rdkit.Chem.MACCSkeys.GenMACCSKeys(chemblMolsArr[ind]))
  else:
    chemblMACCSArr.append(np.zeros(167, dtype=np.int64))
chemblMACCSMat=np.array(chemblMACCSArr)

f=open(featureoutfile+featureoutname+'MACCS.pckl', "wb")
pickle.dump(chemblMACCSMat, f)
f.close()
