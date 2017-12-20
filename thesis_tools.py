import data_tools as dt
import autoencoder_tools as at
import numpy as np

def getIntraparticipantErrors(dataset, autoencoderType): 
    nSpectators = 11

    trainingErrors = []
    testingErrors = []

    dataType = dt.getDataType(dataset)

    for i in range(nSpectators):
        trainingData = dt.getNormalisedData(dataset, "1-25", str(i))
        testingData = dt.getNormalisedData(dataset, "26-30", str(i))

        [autoencoder, _] = at.buildAndTrainAutoencoder(autoencoderType, trainingData, dataType)

        predictedTrainingData = at.predictData(autoencoder, trainingData)
        predictedTestingData = at.predictData(autoencoder, testingData)

        trainingError = dt.getAverageRelativeError(trainingData, predictedTrainingData)
        testingError = dt.getAverageRelativeError(testingData, predictedTestingData)

        trainingErrors += [trainingError]
        testingErrors += [testingError]

    print('')
    print("training set :")
    print("mean : "+str(np.mean(trainingErrors)))
    print("std : "+str(np.std(trainingErrors)))
    print("testing set :")
    print("mean : "+str(np.mean(testingErrors)))
    print("std : "+str(np.std(testingErrors)))
