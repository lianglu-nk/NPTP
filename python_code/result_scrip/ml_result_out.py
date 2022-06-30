import pandas as pd
import itertools
import numpy as np
import pickle
import os
import argparse
basePath=os.getcwd()

def get_file_list(file_folder):
  # method one: file_list = os.listdir(file_folder)
  for root, dirs, file_list in os.walk(file_folder):
    return dirs,file_list
        
parser = argparse.ArgumentParser()
parser.add_argument("-saveBasePath", help="res file path", type=str, default=basePath+'/res_test_data/ML/')
parser.add_argument("-writeDir", help="write file path",type=str, default=basePath+'/results/mlRes/')
parser.add_argument("-datasetNames", help="Dataset Name",type=str, default=["ecfp6fcfp6MACCS"])
parser.add_argument("-methodNames", help="method Name",type=str, default="SVM")
args = parser.parse_args()

saveBasePath = args.saveBasePath
writeDir = args.writeDir
datasetNames = args.datasetNames
methodNames = args.methodNames

if not os.path.exists(writeDir):
  os.makedirs(writeDir)

for method in datasetNames:
  filepath = saveBasePath+method+'/'+methodNames+'/'
  files_list = get_file_list(filepath)[0]
  num = len(files_list)
  AUCall0_0=[]
  AUCall0_1=[]
  AUCall0_2=[]
  AUCmean0=[]
  targets=[]
  
  for i in range(num):
    datapath= filepath + files_list[i]
    targetid = files_list[i]
    f= open(datapath+'/o0001.test.auc.pckl','rb')
    datacontent0_0 = pickle.load(f)
    data0_0=datacontent0_0[0][0]
    f= open(datapath+'/o0002.test.auc.pckl','rb')
    datacontent0_1 = pickle.load(f)
    data0_1=datacontent0_1[0][0]
    f= open(datapath+'/o0003.test.auc.pckl','rb')
    datacontent0_2 = pickle.load(f)
    data0_2=datacontent0_2[0][0]
    AUCall0_0.append(data0_0)
    AUCall0_1.append(data0_1)
    AUCall0_2.append(data0_2)
    mean0=(data0_0+data0_1+data0_2)/3

    AUCmean0.append(mean0)
    targets.append(targetid)
    
  res1={'targets':targets,method+'auc_0':AUCall0_0,method+'auc_1':AUCall0_1,method+'auc_2':AUCall0_2}
  res1_data = pd.DataFrame(res1)
  res2={'targets':targets,method+'AUCmean':AUCmean0}
  res2_data = pd.DataFrame(res2)
  res1_data.to_csv(writeDir+method+'.roc.csv',index=0)
  res2_data.to_csv(writeDir+method+'.rocmean.csv',index=0)
