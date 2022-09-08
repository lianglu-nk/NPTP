
saveFilename=savePrefix+".trainInfo.pckl"
if os.path.isfile(saveFilename):
  saveFile=open(saveFilename, "rb")
  startEpoch=pickle.load(saveFile)
  minibatchCounterTrain=pickle.load(saveFile)
  minibatchCounterTest=pickle.load(saveFile)
  minibatchReportNr=pickle.load(saveFile)
  saveFile.close()

saveFilename=savePrefix+".trainModel.meta"
if os.path.isfile(saveFilename):
  saveFilename=savePrefix+".trainModel"
  with model._get_tf("Graph").as_default():
    tf.train.Saver().restore(model.session, saveFilename)
