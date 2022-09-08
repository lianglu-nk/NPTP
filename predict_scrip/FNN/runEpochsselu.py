currentLR = hyperParams.iloc[paramNr].learningRate 
currentDropout = hyperParams.iloc[paramNr].dropout
currentIDropout = hyperParams.iloc[paramNr].idropout
currentL1Penalty = hyperParams.iloc[paramNr].l1Penalty
currentL2Penalty = hyperParams.iloc[paramNr].l2Penalty
currentMom = hyperParams.iloc[paramNr].mom

reportTrainAUC=[]
reportTrainAP=[]
reportTestAUC=[]
reportTestAP=[]
sumTrainAUC=np.zeros(nrOutputTargets)
sumTrainAUC[:]=np.nan
sumTrainAP=np.zeros(nrOutputTargets)
sumTrainAP[:]=np.nan
sumTestAUC=np.zeros(nrOutputTargets)
sumTestAUC[:]=np.nan
sumTestAP=np.zeros(nrOutputTargets)
sumTestAP[:]=np.nan
startEpoch=0
minibatchCounterTrain=0
minibatchCounterTest=0
minibatchReportNr=0


exec(open(modelScript).read(), globals())
session.run(init)
session.run(biasInitOp, feed_dict={biasInit: trainBias.astype(np.float32)})
if (normalizeGlobalSparse or normalizeLocalSparse) and (nrSparseFeatures > 0.5):
    session.run(sparseMeanInitOp, feed_dict={sparseMeanInit: trainSparseDiv2.reshape(1, -1)})
    session.run(sparseMeanWSparseOp.op)

if continueComputations and loadScript!="":
    exec(open(loadScript).read(), globals())
endEpoch=nrEpochs

if nrSparseFeatures>0.5:
    session.run(sparseMeanWSparseOp.op)


if not np.any(session.run(checkNA)):
    session.run(scalePredictId, feed_dict={inputDropout: currentIDropout, hiddenDropout: currentDropout})
    session.run(scalePredictHd, feed_dict={inputDropout: currentIDropout, hiddenDropout: currentDropout})
    if computeTestPredictions:
        predDenseTest = []
        myfeedDict = {
                        inputDropout: 0.0,
                        hiddenDropout: 0.0,
                        is_training: False
                    }
        if nrDenseFeatures > 0.5:
            batchDenseX = testDenseInput
            myfeedDict.update({xDenseData: batchDenseX})
     
        if useDenseOutputNetPred:
            if compPerformanceTest:
                batchDenseY=testDenseOutput[idxSamplesEval[j]]
            predDenseTest.append(session.run(predNetwork, feed_dict=myfeedDict))
        if useDenseOutputNetPred:
            predDenseTest = np.vstack(predDenseTest)
    session.run(scaleTrainId, feed_dict={inputDropout: currentIDropout, hiddenDropout: currentDropout})
    session.run(scaleTrainHd, feed_dict={inputDropout: currentIDropout, hiddenDropout: currentDropout})

print(predDenseTest.shape)
