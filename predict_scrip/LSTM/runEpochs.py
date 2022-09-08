
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

if continueComputations and loadScript!="":
  exec(open(loadScript).read(), globals())
endEpoch=nrEpochs

if computeTestPredictions:
  predDenseTest=[]
  batchX=np.array([myOneHot(x, oneHot, otherInd, pad_len=seq_length) for x in testSmilesLSTMInput])
  predDenseTest.append(model.predict_on_batch(batchX))
        
  predDenseTest=np.vstack(predDenseTest)
