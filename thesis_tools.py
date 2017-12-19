import data_tools as dt
import autoencoder_tools as at
import numpy as np

def findBestHiddenLayerSizeAide(datasetNames, inputDataType, autoencoderLearningConvergence, autoencoderMaxEpochs, maxAvgRelErr):
    bestHiddenLayerSize = 1
    for datasetName in datasetNames:
        if inputDataType == 0:
            [data, targetData] = dt.getNormalisedClassicalData(datasetName)
        else:
            [data, targetData] = dt.getNormalisedAlternativeData(datasetName)

        # test if the current best hidden layer is satisfactory to the dataset, if such is the case move to the next dataset
        [autoencoder, _] = at.buildAndTrainAutoencoder(bestHiddenLayerSize, data, targetData, autoencoderLearningConvergence, autoencoderMaxEpochs)

        predictedData = autoencoder.predict(data)

        avgRelErr = dt.getAverageRelativeError(data, predictedData)

        if avgRelErr <= maxAvgRelErr:
            print('')
            print(datasetName)
            print(avgRelErr)
            print(bestHiddenLayerSize)
            print('')
            continue

        # test if the maximum number of units has already been reached
        if bestHiddenLayerSize == data.shape[1]:
            print('')
            print(datasetName)
            print(avgRelErr)
            print(bestHiddenLayerSize)
            print('')
            continue

        # perform a binary search to find the lowest number of units in the hidden layer such that the average relative error is lower than or equal to the given maximum relative error
        l = bestHiddenLayerSize+1

        r = data.shape[1]

        while l != r:
            m = (int)(np.floor((l+r)/2))

            [autoencoder, _] = at.buildAndTrainAutoencoder(m, data, targetData, autoencoderLearningConvergence, autoencoderMaxEpochs)

            predictedData = autoencoder.predict(data)

            avgRelErr = dt.getAverageRelativeError(data, predictedData)

            if avgRelErr <= maxAvgRelErr:
                r = m
            else:
                l = m+1

            if l == r:
                print('')
                print(datasetName)
                print(avgRelErr)
                print(l)
                print('')

        bestHiddenLayerSize = l

    return bestHiddenLayerSize

def findBestHiddenLayerSize(datasetNames, inputDataType, autoencoderLearningConvergence, autoencoderMaxEpochs, maxAvgRelErr, nSearches):

    bestHiddenLayerSize = 0

    for i in range(nSearches):
        bestHiddenLayerSize += findBestHiddenLayerSizeAide(datasetNames, inputDataType, autoencoderLearningConvergence, autoencoderMaxEpochs, maxAvgRelErr)

    bestHiddenLayerSize = (int)(np.round(bestHiddenLayerSize/nSearches))

    return bestHiddenLayerSize
