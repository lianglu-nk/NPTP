import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import pandas as pd
import os
import argparse
basePath=os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("-deepDir", help="write file path",type=str, default=basePath+'/results/dnnRes/ecfp6fcfp6MACCS.rocmean.csv')
parser.add_argument("-mlDir", help="write file path",type=str, default=basePath+'/results/mlRes/ecfp6fcfp6MACCS.rocmean.csv')
parser.add_argument("-writerDir", help="write file path",type=str, default=basePath+'/results/')
args = parser.parse_args()
deepBasePath = args.deepDir
mlBasePath = args.mlDir
writeDir = args.writerDir

file_ML=mlBasePath
file_deep=deepBasePath
file_ML_read=pd.read_csv(file_ML)
file_deep_read=pd.read_csv(file_deep)
#AUC#
FNN0=file_deep_read['mean'].tolist()
'''LSTM0=file_deep_read['lstm_ecfp6fcfp6MACCSrocmean'].tolist()
GC0=file_deep_read['graphConvrocmean'].tolist()
Weave0=file_deep_read['graphWeaverocmean'].tolist()'''

SVM0=file_ML_read['ecfp6fcfp6MACCSAUCmean'].tolist()
'''RF0=file_ML_read['ECFP6FCFP6MACCSRFmeanECFP6FCFP6MACCSAUCmean'].tolist()
XGBOOST0=file_ML_read['ECFP6FCFP6MACCSXGBOOSTmeanECFP6FCFP6MACCSAUCmean'].tolist()
KNN0=file_ML_read['ECFP6FCFP6MACCSKNNmeanECFP6FCFP6MACCSAUCmean'].tolist()
NB0=file_ML_read['ECFP6FCFP6MACCSNBmeanECFP6FCFP6MACCSAUCmean'].tolist()'''

#data_AUC = [SVM0,RF0,FNN0,XGBOOST0,GC0,Weave0,KNN0,NB0,LSTM0]
#labels = ['SVM','RF','FNN','XGBOOST','GC','Weave','KNN','NB','LSTM']

data_AUC = [SVM0,FNN0]
labels = ['SVM','FNN']

fig,axs=plt.subplots(nrows=1,ncols=1,figsize=(12, 8))
'''
bplot0=axs.boxplot(data_AUC,patch_artist=True,labels=labels,positions=[1,2,3,4,5,6,7,8,9])
colors = ['DarkBlue', 'SeaGreen','DarkBlue', 'SeaGreen','DarkBlue', 'SeaGreen','DarkBlue', 'SeaGreen','DarkBlue']
'''
bplot0=axs.boxplot(data_AUC,patch_artist=True,labels=labels,positions=[1,2])
colors = ['DarkBlue', 'SeaGreen']
for patch, color in zip(bplot0['boxes'], colors):
        patch.set_facecolor(color)

axs.set_xticklabels(labels,rotation = 90)
ticks = axs.set_yticks([0,0.2,0.4,0.6,0.8,1]) 
#plt.tick_params(labelsize=10)
axs.yaxis.grid(True)
axs.set_xlabel('')
axs.set_ylabel('AUC',fontsize=16)
#y_major_locator=MultipleLocator(0.2)
#plt.ylim(0,)
axs.set_ylim([0,1])
plt.savefig(writeDir+'box_figure.png',dpi=150)
plt.show()
