
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
