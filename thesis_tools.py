import data_tools as dt
import autoencoder_tools as at
import numpy as np

def getIntraparticipantErrorsAid(dataset, autoencoderType): 
    nStimuli = dt.getNumberOfStimuli(dataset)
    nSpectators = dt.getNumberOfSpectators(dataset)

    trainingErrors = []
    testingErrors = []

    dataType = dt.getDataType(dataset)

    for i in range(nSpectators):
        [trainingData, testingData] = dt.getData(dataset, "1-"+str(nStimuli), str(i), 0.83)
        
        [autoencoder, _] = at.buildAndTrainAutoencoder(autoencoderType, trainingData, dataType)

        predictedTrainingData = at.predictData(autoencoder, trainingData)
        predictedTestingData = at.predictData(autoencoder, testingData)

        trainingError = dt.getAverageError(trainingData, predictedTrainingData)
        testingError = dt.getAverageError(testingData, predictedTestingData)

        trainingErrors += [trainingError]
        testingErrors += [testingError]

    return [np.mean(trainingErrors), np.std(trainingErrors), np.mean(testingErrors), np.std(testingErrors)]

def getIntraparticipantErrors(dataset, autoencoderType, nTrials):
    meanTrainingErrors = np.zeros(nTrials)
    stdTrainingErrors = np.zeros(nTrials)
    meanTestingErrors = np.zeros(nTrials)
    stdTestingErrors = np.zeros(nTrials)
    
    for i in range(nTrials):
        [meanTrainingErrors[i], stdTrainingErrors[i], meanTestingErrors[i], stdTestingErrors[i]] = getIntraparticipantErrorsAid(dataset, autoencoderType)

    print('\nOver '+str(nTrials)+' trials, the results for dataset '+dataset+' are as follows :')
    print('Training Errors :')
    print('Mean of the means : '+str(np.mean(meanTrainingErrors)))
    print('Standard deviation of the means : '+str(np.std(meanTrainingErrors)))
    print('Mean of the standard deviations : '+str(np.mean(stdTrainingErrors)))
    print('Standard deviation of ther standard deviations :  '+str(np.std(stdTrainingErrors)))
    print('Testing Errors :')
    print('Mean of the means : '+str(np.mean(meanTestingErrors)))
    print('Standard deviation of the means : '+str(np.std(meanTestingErrors)))
    print('Mean of the standard deviations : '+str(np.mean(stdTestingErrors)))
    print('Standard deviation of ther standard deviations :  '+str(np.std(stdTestingErrors)))

def getInterparticipantErrorsAid(dataset, autoencoderType): 
    nStimuli = dt.getNumberOfStimuli(dataset)
    nSpectators = dt.getNumberOfSpectators(dataset)

    testingErrors = []

    dataType = dt.getDataType(dataset)

    for i in range(nSpectators):
        [trainingData, _] = dt.getData(dataset, "1-"+str(nStimuli), str(i), 1)
        
        [autoencoder, _] = at.buildAndTrainAutoencoder(autoencoderType, trainingData, dataType)

        for j in range(nSpectators):
            if j == i:
                continue

            [testingData, _] = dt.getData(dataset, "1-"+str(nStimuli), str(j), 1)

            predictedTestingData = at.predictData(autoencoder, testingData)

            testingError = dt.getAverageError(testingData, predictedTestingData)

            testingErrors += [testingError]

    return [np.mean(testingErrors), np.std(testingErrors)]

def getInterparticipantErrors(dataset, autoencoderType, nTrials):
    meanTestingErrors = np.zeros(nTrials)
    stdTestingErrors = np.zeros(nTrials)
    
    for i in range(nTrials):
        [meanTestingErrors[i], stdTestingErrors[i]] = getInterparticipantErrorsAid(dataset, autoencoderType)

    print('\nOver '+str(nTrials)+' trials, the results for dataset '+dataset+' are as follows :')
    print('Testing Errors :')
    print('Mean of the means : '+str(np.mean(meanTestingErrors)))
    print('Standard deviation of the means : '+str(np.std(meanTestingErrors)))
    print('Mean of the standard deviations : '+str(np.mean(stdTestingErrors)))
    print('Standard deviation of ther standard deviations :  '+str(np.std(stdTestingErrors)))
