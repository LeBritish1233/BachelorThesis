import scipy.io as sio
import numpy as np

def getNumberOfStimuli(filename):
    if filename[-1] == "r":
        return 10
    return 30

def getNumberOfSpectators(filename):
    return 11

def getDataType(filename):
    if filename[0:3] == "ACC":
        return 0
    return 1

def parseArgument(arg):
    splitArg = list(map(int, arg.split("-")))

    parsedArg = []
    
    for i in range(splitArg[0]-1, splitArg[-1]):
        parsedArg += [i]

    return parsedArg

def normaliseData(data):
    # find the number of stimuli and spectators
    nStimuli = data.shape[1]
    nSpectators = data[0, 0].shape[1]

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
    
    return data

def extractData(data, stimuli, spectators):
    # find the size of the desired normalised data
    nRows = 0
    nCols = data[0, 0][0, 0].shape[1]
    for i in stimuli:
        for j in spectators:
            nRows += data[0, i][0, j].shape[0]


    # copy the data in the normalised data matrix
    extractedData = np.zeros((nRows, nCols))
    rowCounter = 0
    
    for i in stimuli:
        for j in spectators:
            for k in range(data[0, i][0, 0].shape[0]):
                # add the row to the normalised data matrix
                extractedData[rowCounter, :] = data[0, i][0, j][k, :]
                rowCounter += 1

    return extractedData

def getData(filename, stimuli, spectators, trainingProportion):
    # parse the arguments
    stimuli = parseArgument(stimuli)
    spectators = parseArgument(spectators)

    trainingStimuli = stimuli[:(int)(np.round(trainingProportion*len(stimuli)))]
    testingStimuli = stimuli[(int)(np.round(trainingProportion*len(stimuli))):]

    # get the data from the specified file
    data = sio.loadmat('data/'+filename+'.mat')
    data = data[filename]

    # normalise the data
    data = normaliseData(data)

    trainingData = extractData(data, trainingStimuli, spectators)
    testingData = extractData(data, testingStimuli, spectators)

    return [trainingData, testingData]

def getAverageError(data, predictedData):
    avgRelErr = 0
    for i in range(data.shape[0]):
        avgRelErr += np.linalg.norm(data[i, :]-predictedData[i, :])/np.linalg.norm(data[i, :])

    avgRelErr /= data.shape[0]

    return avgRelErr
