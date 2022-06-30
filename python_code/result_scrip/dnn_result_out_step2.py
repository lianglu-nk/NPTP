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
import imp
import os
import sys
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = ''
gpu_options = tf.GPUOptions(allow_growth=True)
import time
import gc
import argparse


basePath=os.getcwd()

np.set_printoptions(threshold=1000)
np.set_printoptions(linewidth=160)
np.set_printoptions(precision=4)
np.set_printoptions(edgeitems=15)
np.set_printoptions(suppress=True)
pd.set_option('display.width', 160)
pd.options.display.float_format = '{:.2f}'.format

parser = argparse.ArgumentParser()
parser.add_argument("-saveBasePath", help="res file path", type=str, default=basePath+'/res_test_data/')
parser.add_argument("-dataPathSave", help="original file path", type=str, default=basePath+'/test_data/')
parser.add_argument("-writeDir", help="write file path",type=str, default=basePath+'/results/dnnRes/')
parser.add_argument("-datasetNames", help="Dataset Name",type=str, default=["ecfp6fcfp6MACCS"])

args = parser.parse_args()
saveBasePath = args.saveBasePath
dataPathSave = args.dataPathSave
writeDir = args.writeDir
datasetNames = args.datasetNames


for datasetName in datasetNames:
  dfROC=pd.read_csv(writeDir+datasetName+".roc.csv",header=0,index_col=0)
  dfROC["mean"]=dfROC[["fold0","fold1","fold2"]].mean(axis=1)
  dfROC=dfROC.drop(["fold0","fold1","fold2"],axis=1)
  
  dfROC.to_csv(writeDir+datasetName+".rocmean.csv")
  
  
