
import rdkit

reportTrainAUC=[]
reportTrainAP=[]
reportTestAUC=[]
reportTestAP=[]
startEpoch=0
minibatchCounterTrain=0
minibatchCounterTest=0
minibatchReportNr=0

exec(open(modelScript).read(), globals())
currentLR=hyperParams.iloc[paramNr].learningRate

if continueComputations and loadScript!="":
    exec(open(loadScript).read(), globals())
endEpoch=nrEpochs

uranium=singleFunc(rdkit.Chem.MolFromSmiles("[U]"))

if computeTestPredictions:
    if smiles_number>128:
        predDenseTest=[]
        idxSamplesEval=[arr[1] for arr in sklearn.model_selection.KFold(n_splits=int(math.ceil(len(testSamples)/batchSize)), shuffle=False).split(np.arange(len(testSamples)))]
        for j in range(len(idxSamplesEval)):
            batchGraphX=mychemblConvertedMols[testGraphInput[idxSamplesEval[j]]]
            batchGraphX=np.array([uranium if type(x)==np.ndarray else x for x in batchGraphX])
            extendSize=0
            if len(batchGraphX)<model.batch_size:
                extendSize=model.batch_size-len(batchGraphX)
                batchGraphX=np.append(batchGraphX, batchGraphX[0:extendSize])
  
            #batchInputSingle=[singleFunc(molX) for molX in batchGraphX]
            batchInput=batchFunc(model, batchGraphX)
            myfeedDict=batchInput
            myfeedDict[model._training_placeholder]=0.0
            with model._get_tf("Graph").as_default():
                predDenseTest.append(model.session.run(model.outputs[0], feed_dict=myfeedDict)[0:(model.batch_size-extendSize)])
    else:
        predDenseTest=[]
        batchGraphX=mychemblConvertedMols[testGraphInput]
        batchGraphX=np.array([uranium if type(x)==np.ndarray else x for x in batchGraphX])
        extendSize=0
        if len(batchGraphX)<model.batch_size:
            extendSize=model.batch_size-len(batchGraphX)
            batchGraphX=np.append(batchGraphX, batchGraphX[0:extendSize])
        #batchInputSingle=[singleFunc(molX) for molX in batchGraphX]
        batchInput=batchFunc(model, batchGraphX)
        myfeedDict=batchInput
        myfeedDict[model._training_placeholder]=0.0
        with model._get_tf("Graph").as_default():
            predDenseTest.append(model.session.run(model.outputs[0], feed_dict=myfeedDict)[0:(model.batch_size-extendSize)])
predDenseTest=np.vstack(predDenseTest)
  
