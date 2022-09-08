
saveFilename = savePrefix + ".trainInfo.pckl"
if os.path.isfile(saveFilename):
    saveFile = open(saveFilename, "rb")
    startEpoch = pickle.load(saveFile)
    minibatchCounterTrain = pickle.load(saveFile)
    minibatchCounterTest = pickle.load(saveFile)
    minibatchReportNr = pickle.load(saveFile)
    saveFile.close()

saveFilename = savePrefix + ".trainModel.meta"
if os.path.isfile(saveFilename):
    saveFilename = savePrefix + ".trainModel"
    tf.train.Saver().restore(session, saveFilename)
