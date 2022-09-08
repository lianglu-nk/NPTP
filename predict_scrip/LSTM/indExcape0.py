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
from keras.models import load_model
gpu_options=tf.ConfigProto()
gpu_options.gpu_options.allow_growth=True

import time
import gc
import argparse
import utilsLib
import actLib
import rdkit
from rdkit.Chem import MACCSkeys
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import AllChem
from rdkit import DataStructs

basePath=os.getcwd()
scrip_path=basePath+"/predict_scrip/LSTM/"

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
parser.add_argument("-method_name", help="method name", type=str, default='LSTM')
parser.add_argument("-datatset_type", help="dataset type", type=str, default='chembl29')
parser.add_argument("-saveBasePath", help="saveBasePath", type=str, default=basePath+'/result/')
args = parser.parse_args()
availableGPUs = args.availableGPUs
smiles_file = args.smiles_file
method_name = args.method_name
datatset_type = args.datatset_type
saveBasePath = args.saveBasePath
dataPathSave1=basePath+'/model/'+datatset_type+'/'+datatset_type+'_original_data/'
dataSave=basePath+'/model/'+datatset_type+'/'+'lstm_ecfp6fcfp6MACCS/'
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
#SMILES to fingerprint feature conversion
smiles_list=file_read['smiles'].tolist()
chemblMolsArr=[]
chemblSmilesArr=[]
chemblMACCSArr=[]
smiles_number=len(smiles_list)
ecfpMat = np.zeros((smiles_number, 2048), dtype=int)
fcfpMat = np.zeros((smiles_number, 2048), dtype=int)
for i in range(smiles_number):
    mol_id=rdkit.Chem.MolFromSmiles(smiles_list[i])
    fp = AllChem.GetMorganFingerprintAsBitVect(mol_id, 3, 2048)
    fp_1 = AllChem.GetMorganFingerprintAsBitVect(mol_id, 3, 2048, useFeatures=True)
    ecfpMat[i] = np.array(list(fp.ToBitString()))
    fcfpMat[i] = np.array(list(fp_1.ToBitString()))
    chemblMACCSArr.append(rdkit.Chem.MACCSkeys.GenMACCSKeys(mol_id))
    chemblMolsArr.append(mol_id)
    chemblSmilesArr.append(smiles_list[i])
ecfpMat = scipy.sparse.csr_matrix(ecfpMat)
fcfpMat = scipy.sparse.csr_matrix(fcfpMat)
rdkitArr=np.array(chemblMolsArr)
maccsMat=np.array(chemblMACCSArr)
sampleECFPInd = pd.Series(data=np.arange(len(smiles_list)), index=np.arange(len(smiles_list)))


maccsInputData = maccsMat
ecfpInputData = ecfpMat
fcfpInputData = fcfpMat
smilesArr = chemblSmilesArr


denseInputData = None
denseSampleIndex = None
sparseInputData = None
sparseSampleIndex = None
lstmGraphInputData = rdkitArr
lstmSmilesInputData = smilesArr
lstmSampleIndex = pd.Series(np.arange(smiles_number))



allSamples=np.array([], dtype=np.int64)
if not (lstmSmilesInputData is None):
  allSamples=np.union1d(allSamples, lstmSampleIndex.index.values)
if not (lstmSmilesInputData is None):
  allSamples=np.intersect1d(allSamples, lstmSampleIndex.index.values)
allSamples=allSamples.tolist()
testSamples=list(set(allSamples))

nrDenseFeatures=0
nrSparseFeatures=0

if ("lstmSmilesInputData" in globals()) and (not (lstmSmilesInputData is None)):
    testSmilesLSTMInput = np.array(lstmSmilesInputData)[lstmSampleIndex[testSamples].values].copy()
    testGrtaphLSTMInput = np.array(lstmGraphInputData)[lstmSampleIndex[testSamples].values].copy()


    testLSTMSideOutputMACCS = maccsInputData[lstmSampleIndex[testSamples].values]
    testLSTMSideOutputECFP = ecfpInputData[lstmSampleIndex[testSamples].values]
    testLSTMSideOutputFCFP = fcfpInputData[lstmSampleIndex[testSamples].values]
    #Output bits with frequent features
    testLSTMSideOutputECFP = testLSTMSideOutputECFP[:, np.argsort(-testLSTMSideOutputECFP.sum(0).A[0])[0:128]].A
    testLSTMSideOutputFCFP = testLSTMSideOutputFCFP[:, np.argsort(-testLSTMSideOutputFCFP.sum(0).A[0])[0:128]].A

    testLSTMSideOutput = np.hstack([testLSTMSideOutputECFP, testLSTMSideOutputFCFP, testLSTMSideOutputMACCS])
#Names of different models
if datatset_type in ['chembl29']:
    savePrefix0='noweakchembl29endmodel'
elif datatset_type in ['chembl26']:
    savePrefix0='noweakchembl26end'
elif datatset_type in ['derivant_chembl29']:
    savePrefix0='noweakderivantendmodel'
else:
    savePrefix0='noweakderivant_allend'
savePrefix=dataSave+savePrefix0
#Load model 
modelScript=scrip_path+'models.py'


loadScript=scrip_path+'step2Load.py'
saveScript=""

exec(open(scrip_path+'runEpochs.py').read(), globals())
predMatrix=pd.DataFrame(data=predDenseTest, index=testSamples)
num=np.arange(0,nrOutputTargets).tolist()
predMatrix=predMatrix.iloc[:,num]
predMatrix.columns=targetAnnInd.index.values.tolist()
print(predMatrix)
predMatrix.to_csv(saveBasePath+datatset_type+method_name+'.csv', index=True, header=True)
