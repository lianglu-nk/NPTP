
import rdkit

reportTrainAUC=[]
reportTrainAP=[]
reportTestAUC=[]
reportTestAP=[]
startEpoch=0
minibatchCounterTrain=0
minibatchCounterTest=0
minibatchReportNr=0


basicArchitecture=hyperParams.iloc[paramNr].basicArchitecture
currentLR=hyperParams.iloc[paramNr].learningRate
dropout=hyperParams.iloc[paramNr].dropout
graphLayers=hyperParams.iloc[paramNr].graphLayers
denseLayers=hyperParams.iloc[paramNr].denseLayers


import deepchem.feat
import deepchem.feat.mol_graphs
import deepchem.metrics

exec(open(scrip_path+'graphModels.py').read(), globals())

seed=123
if basicArchitecture=="GraphConv":
    model=MyGraphConvTensorGraph(n_tasks=nrOutputTargets, graph_conv_layers=graphLayers, dense_layer_size=denseLayers, batch_size=batchSize, dropout=dropout, learning_rate=currentLR, use_queue=True, random_seed=seed, mode='classification', configproto=gpu_options, verbose=False, mycapacity=5)
    singleFunc=convFunc
    batchFunc=convInput

if not model.built:
    model.build()
    with model._get_tf("Graph").as_default():
        updateOps=tf.get_collection(tf.GraphKeys.UPDATE_OPS)

currentLR=hyperParams.iloc[paramNr].learningRate

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
  
