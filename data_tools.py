import scipy.io as sio
import numpy as np

def getNormalisedData(filename):
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
        for j in range(nSpectators):
            nRows += data[0, i][0, j].shape[0]
            
    # initiate the normalised data matrix and initiate a row counter for data copying
    normalisedData = np.zeros((nRows, nCols))
    rowCounter = 0
    minData = np.inf
    maxData = -np.inf

    for i in range(nStimuli):
        for j in range(nSpectators):
            for k in range(data[0, i][0, j].shape[0]):
                row = data[0, i][0, j][k, :]
                minRow = min(row)
                maxRow = max(row)
                if minRow < minData:
                    minData = minRow
                if maxRow > maxData:
                    maxData = maxRow

    # copy the data in the normalised data matrix
    for i in range(nStimuli):
        for j in range(nSpectators):
            for k in range(data[0, i][0, j].shape[0]):
                # get the row and find its minimum and maximum
                row = data[0, i][0, j][k, :]

                # normalise the row
                row = (row - minData)/(maxData - minData)

                # add the row to the normalised data matrix
                normalisedData[rowCounter, :] = row
                rowCounter += 1

    return normalisedData

def getAverageRelativeError(data, predictedData):
    avgRelErr = 0
    for i in range(data.shape[0]):
        avgRelErr += np.linalg.norm(data[i, :]-predictedData[i, :])/np.linalg.norm(data[i, :])

    avgRelErr /= data.shape[0]

    return avgRelErr
