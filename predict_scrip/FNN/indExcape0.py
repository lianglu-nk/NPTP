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

os.environ['CUDA_VISIBLE_DEVICES'] = ''
gpu_options = tf.GPUOptions(allow_growth=True)
import time
import gc
import argparse
import actLib
import rdkit
from rdkit.Chem import MACCSkeys
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import AllChem
from rdkit import DataStructs
import utilsLib

basePath=os.getcwd()
scrip_path=basePath+"/predict_scrip/FNN/"

# np.set_printoptions(threshold='nan')
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
parser.add_argument("-method_name", help="method name", type=str, default='FNN')
parser.add_argument("-datatset_type", help="dataset type", type=str, default='chembl29')
parser.add_argument("-saveBasePath", help="saveBasePath", type=str, default=basePath+'/result/')
args = parser.parse_args()
availableGPUs = args.availableGPUs
smiles_file = args.smiles_file
method_name = args.method_name
datatset_type = args.datatset_type
saveBasePath = args.saveBasePath
dataPathSave1=basePath+'/model/'+datatset_type+'/'+datatset_type+'_original_data/'
dataSave=basePath+'/model/'+datatset_type+'/'+'ecfp6fcfp6MACCS/'
sparseOutputData=None
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
nrSparseFeatures=0
trainBias=np.zeros(nrOutputTargets)

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
#SMILES to fingerprint feature conversion
smiles_list=file_read['smiles'].tolist()
MACCSArray = []
smiles_number=len(smiles_list)
ecfpMat = np.zeros((smiles_number, 2048), dtype=int)
fcfpMat = np.zeros((smiles_number, 2048), dtype=int)
for i in range(smiles_number):
    mol_id=rdkit.Chem.MolFromSmiles(smiles_list[i])
    fp = AllChem.GetMorganFingerprintAsBitVect(mol_id, 3, 2048)
    fp_1 = AllChem.GetMorganFingerprintAsBitVect(mol_id, 3, 2048, useFeatures=True)
    ecfpMat[i] = np.array(list(fp.ToBitString()))
    fcfpMat[i] = np.array(list(fp_1.ToBitString()))
    MACCSArray.append(MACCSkeys.GenMACCSKeys(mol_id))
MACCSMat=np.array(MACCSArray)
sampleECFPInd = pd.Series(data=np.arange(len(smiles_list)), index=np.arange(len(smiles_list)))

denseInputData = np.hstack([ecfpMat, fcfpMat, MACCSMat])
denseSampleIndex = sampleECFPInd
sparseInputData = None
sparseSampleIndex = None

del ecfpMat
del sampleECFPInd
del fcfpMat
del MACCSMat
allSamples = np.array([], dtype=np.int64)
if not (denseInputData is None):
    allSamples = np.union1d(allSamples, denseSampleIndex.index.values)
if not (sparseInputData is None):
    allSamples = np.union1d(allSamples, sparseSampleIndex.index.values)
if not (denseInputData is None):
    allSamples = np.intersect1d(allSamples, denseSampleIndex.index.values)
if not (sparseInputData is None):
    allSamples = np.intersect1d(allSamples, sparseSampleIndex.index.values)
allSamples = allSamples.tolist()
testSamples = list(set(allSamples))
  

#Fingerprint characteristics of training dataset
f = open(dataPathSave1 + 'testecfp6dense.pckl', "rb")
ecfpMat = pickle.load(f)
sampleECFPInd = pickle.load(f)
# featureECFPInd=pickle.load(f)
f.close()

f = open(dataPathSave1 + 'testfcfp6dense.pckl', "rb")
fcfpMat = pickle.load(f)
sampleFCFPInd = pickle.load(f)
# featureSemiInd=pickle.load(f)
f.close()

f = open(dataPathSave1 + 'testMACCSFNNdense.pckl', "rb")
MACCSMat = pickle.load(f)
sampleMACCSInd = pickle.load(f)
# featureToxInd=pickle.load(f)
f.close()

denseInputData1 = np.hstack([ecfpMat, fcfpMat, MACCSMat])
denseSampleIndex1 = sampleECFPInd
sparseInputData = None
sparseSampleIndex = None

del ecfpMat
del sampleECFPInd
del fcfpMat
del sampleFCFPInd
del MACCSMat
del sampleMACCSInd
allSamples1 = np.array([], dtype=np.int64)
if not (denseInputData is None):
    allSamples1 = np.union1d(allSamples1, denseSampleIndex1.index.values)
if not (sparseInputData is None):
    allSamples1 = np.union1d(allSamples1, sparseSampleIndex1.index.values)
if not (denseInputData is None):
    allSamples1 = np.intersect1d(allSamples1, denseSampleIndex1.index.values)
if not (sparseInputData is None):
    allSamples1 = np.intersect1d(allSamples1, sparseSampleIndex1.index.values)
allSamples1 = allSamples1.tolist()
trainSamples = list(set(allSamples1))

normalizeGlobalDense = False
normalizeGlobalSparse = False
normalizeLocalDense = False
normalizeLocalSparse = False
if not denseInputData is None:
    normalizeLocalDense = True
if not sparseInputData is None:
    normalizeLocalSparse = True

nrDenseFeatures = 0
if not (denseInputData is None):
    trainDenseInput = denseInputData1[denseSampleIndex1[trainSamples].values].copy()
    testDenseInput = denseInputData[denseSampleIndex[testSamples].values].copy()
    nrDenseFeatures = trainDenseInput.shape[1]
#Normalized
    if normalizeLocalDense:
        trainDenseMean1 = np.nanmean(trainDenseInput, 0)
        trainDenseStd1 = np.nanstd(trainDenseInput, 0) + 0.0001
        trainDenseInput = (trainDenseInput - trainDenseMean1) / trainDenseStd1
        trainDenseInput = np.tanh(trainDenseInput)
        trainDenseMean2 = np.nanmean(trainDenseInput, 0)
        trainDenseStd2 = np.nanstd(trainDenseInput, 0) + 0.0001
        trainDenseInput = (trainDenseInput - trainDenseMean2) / trainDenseStd2

        testDenseInput = (testDenseInput - trainDenseMean1) / trainDenseStd1
        testDenseInput = np.tanh(testDenseInput)
        testDenseInput = (testDenseInput - trainDenseMean2) / trainDenseStd2

    trainDenseInput = np.nan_to_num(trainDenseInput)
    testDenseInput = np.nan_to_num(testDenseInput)
    


exec(open(scrip_path + 'hyperparams.py').read(), globals())
#Optimal parameter index corresponding to different models
if datatset_type in ['chembl29']:
    paramNr =24
elif datatset_type in ['chembl26']:
    paramNr =24
elif datatset_type in ['derivant_chembl29']:
    paramNr =24
else:
    paramNr =2
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
basicArchitecture=[hyperParams.iloc[paramNr].basicArchitecture]
print(basicArchitecture)
#Load model 
modelScript=scrip_path+'model'+basicArchitecture[0]+'.py'
loadScript=scrip_path+'step2Load.py'
saveScript=""

exec(open(scrip_path+'runEpochs'+basicArchitecture[0]+'.py').read(), globals())
predMatrix=pd.DataFrame(data=predDenseTest, index=testSamples, columns=targetAnnInd.index.values.tolist())
print(predMatrix)
predMatrix.to_csv(saveBasePath+datatset_type+method_name+'.csv', index=True, header=True)
