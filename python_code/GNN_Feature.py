#Parts of code in this file have been taken (copied) from https://github.com/ml-jku/lsc
#Copyright (C) 2018 Andreas Mayr

from __future__ import print_function
from __future__ import division
import math
import itertools
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.sparse
import sklearn
import sklearn.feature_selection
import sklearn.model_selection
import sklearn.metrics
import h5py
import pickle
import imp
import os
import sys
import time
import gc
import rdkit
import rdkit.Chem
import deepchem
import deepchem.feat
import concurrent
import concurrent.futures 
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
  if mol is not None:
    mol=rdkit.Chem.rdmolops.RenumberAtoms(mol, rdkit.Chem.rdmolfiles.CanonicalRankAtoms(mol))  
  chemblMolsArr.append(mol)

chemblMolsArrCopy=np.array(chemblMolsArr)
fileObject=open(featureoutfile+featureoutname+"Deepchem.pckl",'wb')
pickle.dump(chemblMolsArrCopy, fileObject)
fileObject.close()

convFeat=deepchem.feat.ConvMolFeaturizer()
weaveFeat=deepchem.feat.WeaveFeaturizer()

def convFunc(x):
  return convFeat([x])[0]

def weaveFunc(x):
  return weaveFeat([x])[0]

weaveConv=[]
for i in range(len(chemblMolsArr)):
  if i%1000==0:
    print(i)
  weaveConv.append(weaveFunc(chemblMolsArr[i]))

convConv=[]
for i in range(len(chemblMolsArr)):
  if i%1000==0:
    print(i)
  convConv.append(convFunc(chemblMolsArr[i]))


convConv=np.array(convConv)
f=open(featureoutfile+featureoutname+'Conv.pckl', "wb")
pickle.dump(convConv, f)
f.close()
  
weaveConv=np.array(weaveConv)
f=open(featureoutfile+featureoutname+'Weave.pckl', "wb")
pickle.dump(weaveConv, f)
f.close()


