
reportTrainAUC = []
reportTrainAP = []
reportTrainF1 = []
reportTrainKAPPA = []
reportTestAUC = []
reportTestAP = []
reportTestF1 = []
reportTestKAPPA = []
startEpoch = 0
minibatchCounterTrain = 0
minibatchCounterTest = 0
minibatchReportNr = 0

exec(open(modelScript).read(), globals())

saveFilename=savePrefix+".trainInfo.pckl"
if os.path.isfile(saveFilename):
  saveFile=open(saveFilename, "rb")
  startEpoch=pickle.load(saveFile)
  minibatchCounterTrain=pickle.load(saveFile)
  minibatchCounterTest=pickle.load(saveFile)
  minibatchReportNr=pickle.load(saveFile)
  saveFile.close()

saveFilename=savePrefix+".trainModel"
if os.path.isfile(saveFilename):
  saveFilename=savePrefix+".trainModel"
  #model=load_model(saveFilename)
  model.load_weights(saveFilename)
endEpoch=nrEpochs

if computeTestPredictions:
  predDenseTest=[]
  batchX=np.array([myOneHot(x, oneHot, otherInd, pad_len=seq_length) for x in testSmilesLSTMInput])
  predDenseTest.append(model.predict_on_batch(batchX))
        
  predDenseTest=np.vstack(predDenseTest)
