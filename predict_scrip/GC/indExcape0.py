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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
#import imp
import os
import sys
import tensorflow as tf
import actLib

os.environ['CUDA_VISIBLE_DEVICES'] = ''
gpu_options=tf.ConfigProto()
gpu_options.gpu_options.allow_growth=True

import time
import gc
import argparse
import rdkit
import rdkit.Chem
import deepchem
import deepchem.feat
import concurrent
import concurrent.futures 

basePath=os.getcwd()
scrip_path=basePath+"/predict_scrip/GC/"

#np.set_printoptions(threshold='nan')
np.set_printoptions(threshold=1000)
np.set_printoptions(linewidth=160)
np.set_printoptions(precision=4)
np.set_printoptions(edgeitems=15)
np.set_printoptions(suppress=True)
pd.set_option('display.width', 160)
pd.options.display.float_format = '{:.2f}'.format
#External parameters input
parser = argparse.ArgumentParser()
parser.add_argument("-availableGPUs", help="Available GPUs", nargs='*', type=int, default=[0])
parser.add_argument("-smiles_file", help="smiles needed to be predicted", type=str, default=basePath+'/data/test.csv')
parser.add_argument("-method_name", help="method name", type=str, default='GC')
parser.add_argument("-datatset_type", help="dataset type", type=str, default='chembl29')
parser.add_argument("-saveBasePath", help="saveBasePath", type=str, default=basePath+'/result/')
args = parser.parse_args()
availableGPUs = args.availableGPUs
smiles_file = args.smiles_file
method_name = args.method_name
datatset_type = args.datatset_type
saveBasePath = args.saveBasePath
dataPathSave1=basePath+'/model/'+datatset_type+'/'+datatset_type+'_original_data/'
dataSave=basePath+'/model/'+datatset_type+'/'+'graphConv/'
batchSize=128
#The number of predicted targets corresponding to different models
if datatset_type in ['chembl29']:
  nrOutputTargets=949
elif datatset_type in ['chembl26']:
  nrOutputTargets=899
elif datatset_type in ['derivant_chembl29']:
  nrOutputTargets=467
else:
  nrOutputTargets=470

normalizeGlobalSparse=False
normalizeLocalSparse=False
continueComputations=True
saveComputations=False
nrEpochs=0
computeTrainPredictions=False
compPerformanceTrain=False
computeTestPredictions=True
compPerformanceTest=False
useDenseOutputNetPred=True
savePredictionsAtBestIter=False
logPerformanceAtBestIter=False
runEpochs=False

availableGPUs=[]
if len(availableGPUs)>0.5:
  os.environ['CUDA_VISIBLE_DEVICES']=str(availableGPUs[0])

file_read=pd.read_csv(smiles_file)

f=open(dataPathSave1+'labelsHard.pckl', "rb")
targetMat=pickle.load(f)
sampleAnnInd=pickle.load(f)
targetAnnInd=pickle.load(f)
f.close()

targetAnnInd=targetAnnInd
targetAnnInd=targetAnnInd-targetAnnInd.min()
#Molecular to graph feature conversion
smiles_list=file_read['smiles'].tolist()
chemblMolsArr=[]
smiles_number=len(smiles_list)
for i in range(smiles_number):
  mol_id=rdkit.Chem.MolFromSmiles(smiles_list[i])
  mol_id=rdkit.Chem.rdmolops.RenumberAtoms(mol_id, rdkit.Chem.rdmolfiles.CanonicalRankAtoms(mol_id))
  chemblMolsArr.append(mol_id)
chemblMolsArrCopy=np.array(chemblMolsArr)
convFeat=deepchem.feat.ConvMolFeaturizer()
def convFunc(x):
  return convFeat([x])[0]
convConv=[]
for i in range(len(chemblMolsArrCopy)):
  convConv.append(convFunc(chemblMolsArrCopy[i]))
convConv=np.array(convConv)
mychemblConvertedMols = convConv
chemblMolsArr = np.arange(len(mychemblConvertedMols))

denseInputData = None
denseSampleIndex = None
sparseInputData = None
sparseSampleIndex = None
graphInputData = chemblMolsArr
#48406
graphSampleIndex = pd.Series(np.arange(smiles_number))

allSamples=np.array([], dtype=np.int64)
if not (graphInputData is None):
  allSamples=np.union1d(allSamples, graphSampleIndex.index.values)
if not (graphInputData is None):
  allSamples=np.intersect1d(allSamples, graphSampleIndex.index.values)
allSamples=allSamples.tolist()

testSamples = list(set(allSamples))

nrDenseFeatures=0
nrSparseFeatures=0

if ("graphInputData" in globals()) and (not (graphInputData is None)):
  testGraphInput=graphInputData[graphSampleIndex[testSamples].values].copy()

dictionary0 = {
  'basicArchitecture': ['GraphConv'],
  'learningRate': [0.001, 0.0001],
  'dropout': [0.0, 0.5],
  'graphLayers': [[128]*2, [128]*3, [128]*4],
  'denseLayers': [[1024], [2048]]
}

hyperParams=pd.DataFrame(list(itertools.product(*dictionary0.values())), columns=dictionary0.keys())
hyperParams.index=np.arange(len(hyperParams.index.values))

#Optimal parameter index corresponding to different models
if datatset_type in ['chembl29']:
  paramNr =18
elif datatset_type in ['chembl26']:
  paramNr =18
elif datatset_type in ['derivant_chembl29']:
  paramNr =18
else:
  paramNr =20
#Names of different models
if datatset_type in ['chembl29']:
  savePrefix0='chembl29model'
elif datatset_type in ['chembl26']:
  savePrefix0='noweakchembl26end'
elif datatset_type in ['derivant_chembl29']:
  savePrefix0='derivantallmodel'
else:
  savePrefix0='noweakderivant_allend'
savePrefix=dataSave+savePrefix0

#Load model
saveScript=""
exec(open(scrip_path+'runEpochs.py').read(), globals())
predMatrix=pd.DataFrame(data=predDenseTest, index=testSamples, columns=targetAnnInd.index.values.tolist())
print(predMatrix)
predMatrix.to_csv(saveBasePath+datatset_type+method_name+'.csv', index=True, header=True)
