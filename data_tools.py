import scipy.io as sio
import numpy as np

def getNormalisedClassicalData(filename):
    # get the data from the specified file
    dataContainer = sio.loadmat('data/'+filename+'.mat')
    data = dataContainer[filename]

    # find the number of stimuli and spectators
    nStimuli = data.shape[1]
    nSpectators = data[0, 0].shape[1]

    # initiate the number of rows of the normalised data matrix and get the number of columns
    nRows = 0
    nCols = data[0, 0][0, 0].shape[1]

    # find the number of rows
    for i in range(nStimuli):
        nRows += nSpectators*data[0, i][0, 0].shape[0]
            
    # find the values for normalisation
    minData = np.inf
    maxData = -np.inf

    for i in range(nStimuli):
        for j in range(nSpectators):
            for k in range(data[0, i][0, 0].shape[0]):
                row = data[0, i][0, j][k, :]
                minRow = min(row)
                maxRow = max(row)
                if minRow < minData:
                    minData = minRow
                if maxRow > maxData:
                    maxData = maxRow
    
    # normalise the data
    data = (data-minData)/(maxData-minData)

    # copy the data in the normalised data matrix
    normalisedData = np.zeros((nRows, nCols))
    rowCounter = 0
    
    for i in range(nStimuli):
        for j in range(nSpectators):
            for k in range(data[0, i][0, 0].shape[0]):
                # add the row to the normalised data matrix
                normalisedData[rowCounter, :] = data[0, i][0, j][k, :]
                rowCounter += 1

    return [normalisedData, normalisedData]

def getNormalisedAlternativeData(filename):
    # get the data from the specified file
    dataContainer = sio.loadmat('data/'+filename+'.mat')
    data = dataContainer[filename]

    # find the number of stimuli and spectators
    nStimuli = data.shape[1]
    nSpectators = data[0, 0].shape[1]

    # initiate the number of rows of the normalised data matrix and get the number of columns
    nRows = 0
    nCols = data[0, 0][0, 0].shape[1]

    # find the number of rows
    for i in range(nStimuli):
        nRows += nSpectators*nSpectators*data[0, i][0, 0].shape[0]
            
    # find the values for normalisation
    minData = np.inf
    maxData = -np.inf

    for i in range(nStimuli):
        for j in range(nSpectators):
            for k in range(data[0, i][0, 0].shape[0]):
                row = data[0, i][0, j][k, :]
                minRow = min(row)
                maxRow = max(row)
                if minRow < minData:
                    minData = minRow
                if maxRow > maxData:
                    maxData = maxRow

    # normalise the data
    data = (data-minData)/(maxData-minData)

    # copy the data in the normalised data matrix
    normalisedData = np.zeros((nRows, nCols))
    rowCounter = 0
    
    for i in range(nStimuli):
        for j in range(nSpectators):
            for k in range(data[0, i][0, 0].shape[0]):
                # add the row to the normalised data matrix by the number of spectators it needs to be mapped to
                for l in range(nSpectators):
                    normalisedData[rowCounter, :] = data[0, i][0, j][k, :]
                    rowCounter += 1

    # copy the data in the normalised target data matrix
    normalisedTargetData = np.zeros((nRows, nCols))
    rowCounter = 0

    for i in range(nStimuli):
        for j in range(nSpectators):
            for k in range(data[0, i][0, 0].shape[0]):
                for l in range(nSpectators):
                    normalisedTargetData[rowCounter, :] = data[0, i][0, l][k, :]
                    rowCounter += 1
    
    return [normalisedData, normalisedTargetData]

def getAverageRelativeError(data, predictedData):
    avgRelErr = 0
    for i in range(data.shape[0]):
        avgRelErr += np.linalg.norm(data[i, :]-predictedData[i, :])/np.linalg.norm(data[i, :])

    avgRelErr /= data.shape[0]

    return avgRelErr
