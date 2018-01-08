import scipy.io as sio
import numpy as np

def getDataType(filename):
    if filename[0:3] == "ACC":
        return 0
    return 1

def normaliseData(data):
    # find the values for normalisation
    minData = np.inf
    maxData = -np.inf

    for i in range(data.shape[1]):
        for j in range(data[0, 0].shape[1]):
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

def extractData(data, stimuli):
    # find the size of the desired normalised data
    nRows = 0
    nCols = data[0, 0][0, 0].shape[1]
    for i in stimuli:
        for j in range(data[0, 0].shape[1]):
            nRows += data[0, i][0, j].shape[0]


    # copy the data in the normalised data matrix
    extractedData = np.zeros((nRows, nCols))
    rowCounter = 0
    
    for i in stimuli:
        for j in range(data[0, 0].shape[1]):
            for k in range(data[0, i][0, 0].shape[0]):
                # add the row to the normalised data matrix
                extractedData[rowCounter, :] = data[0, i][0, j][k, :]
                rowCounter += 1

    return extractedData

def getData(filename, testGroup):
    # get the data from the specified file
    data = sio.loadmat('data/'+filename+'.mat')
    data = data[filename]

    # normalise the data
    data = normaliseData(data)

    # get training and testing stimuli
    trainingStimuli = []
    for i in range(data.shape[1]):
        trainingStimuli += [i]

    groupSize = (int)(data.shape[1]/10)
    testingStimuli = []
    for i in range(groupSize):
        testingStimuli += [groupSize*testGroup+i]

    trainingStimuli = [x for x in trainingStimuli if x not in testingStimuli]

    # extract the data
    trainingData = extractData(data, trainingStimuli)
    testingData = extractData(data, testingStimuli)

    return [trainingData, testingData]

def correlationCoefficient(data, predictedData):
    n = 0
    d1 = 0
    d2 = 0
    md = np.mean(data)
    mpd = np.mean(predictedData)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            n += (data[i, j]-md)*(predictedData[i, j]-mpd)
            d1 += (data[i, j]-md)*(data[i, j]-md)
            d2 += (predictedData[i, j]-mpd)*(predictedData[i, j]-mpd)

    return n/np.sqrt(d1*d2)

def concordanceCorrelationCoefficient(data, predictedData):
    cc = correlationCoefficient(data, predictedData)
    md = np.mean(data)
    mpd = np.mean(predictedData)
    stdd = np.std(data)
    stdpd = np.std(predictedData)

    return 2*cc*stdd*stdpd/(stdd*stdd+stdpd*stdpd+(md-mpd)*(md-mpd))
