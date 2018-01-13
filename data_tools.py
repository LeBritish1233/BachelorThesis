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

def getEncodedData(filename, encoderType, group):
    data = sio.loadmat('data/'+filename+'.mat')
    data = data[filename]
 
    groupSize = (int)(data.shape[1]/10)

    groupStimuli = np.zeros(groupSize, int)
    for i in range(groupSize):
        groupStimuli[i] = groupSize*group+i

    groupStimuliSizes = np.zeros(groupSize, int)
    for i in range(groupSize):
        groupStimuliSizes[i] = data[0, groupStimuli[i]][0, 0].shape[0]

    if 'lag1' in filename:
        labels = sio.loadmat('data/Arousal_all_movies_win5_lag1.mat')
        labels = labels['Arousal_all_movies_win5_lag1']
    else:
        labels = sio.loadmat('data/Arousal_all_movies_win5_lag5.mat')
        labels = labels['Arousal_all_movies_win5_lag5']

    desiredGroupStimuliSizes = np.zeros(groupSize, int)
    for i in range(groupSize):
        desiredGroupStimuliSizes[i] = labels[0, groupStimuli[i]].shape[0]

    encodedData = np.load('model_outputs/'+filename+'_encoder_'+str(encoderType)+'_group_'+str(group)+'.npy')

    reshapedEncodedData = np.zeros((sum(desiredGroupStimuliSizes),data[0, 0].shape[1]*encodedData.shape[1]))

    rc = 0

    for i in range(groupSize):
        for j in range(data[0, 0].shape[1]):
            for k in range(desiredGroupStimuliSizes[i]):
                reshapedEncodedData[k+sum(desiredGroupStimuliSizes[:i]), j*encodedData.shape[1]:(j+1)*encodedData.shape[1]] += encodedData[rc, :]
                rc += 1
                if k == desiredGroupStimuliSizes[i]-1:
                    rc += groupStimuliSizes[i]-desiredGroupStimuliSizes[i]

    return reshapedEncodedData

def getEncodedDataCrossed(filename, crossedFilename, encoderType, group):
    data = sio.loadmat('data/'+filename+'.mat')
    data = data[filename]
 
    groupSize = (int)(data.shape[1]/10)

    groupStimuli = np.zeros(groupSize, int)
    for i in range(groupSize):
        groupStimuli[i] = groupSize*group+i

    groupStimuliSizes = np.zeros(groupSize, int)
    for i in range(groupSize):
        groupStimuliSizes[i] = data[0, groupStimuli[i]][0, 0].shape[0]

    if 'lag1' in filename:
        labels = sio.loadmat('data/Arousal_all_movies_win5_lag1.mat')
        labels = labels['Arousal_all_movies_win5_lag1']
    else:
        labels = sio.loadmat('data/Arousal_all_movies_win5_lag5.mat')
        labels = labels['Arousal_all_movies_win5_lag5']

    desiredGroupStimuliSizes = np.zeros(groupSize, int)
    for i in range(groupSize):
        desiredGroupStimuliSizes[i] = labels[0, groupStimuli[i]].shape[0]

    encodedData = np.load('model_outputs/'+filename+'_crossed_'+crossedFilename+'_encoder_'+str(encoderType)+'_group_'+str(group)+'.npy')

    reshapedEncodedData = np.zeros((sum(desiredGroupStimuliSizes),data[0, 0].shape[1]*encodedData.shape[1]))

    rc = 0

    for i in range(groupSize):
        for j in range(data[0, 0].shape[1]):
            for k in range(desiredGroupStimuliSizes[i]):
                reshapedEncodedData[k+sum(desiredGroupStimuliSizes[:i]), j*encodedData.shape[1]:(j+1)*encodedData.shape[1]] += encodedData[rc, :]
                rc += 1
                if k == desiredGroupStimuliSizes[i]-1:
                    rc += groupStimuliSizes[i]-desiredGroupStimuliSizes[i]

    return reshapedEncodedData

def getRegressionInputData(filename, encoderType, testGroup):
    trainingDataStarted = False
    for i in range(10):
        encodedData = getEncodedData(filename, encoderType, i)
        if i == testGroup:
            testingData = encodedData
        elif trainingDataStarted:
            trainingData = np.concatenate((trainingData, encodedData))
        else:
            trainingData = encodedData
            trainingDataStarted = True
    return [trainingData, testingData]

def getRegressionInputDataCrossed(filename, crossedFilename, encoderType, testGroup):
    trainingDataStarted = False
    for i in range(10):
        encodedData = getEncodedDataCrossed(filename, crossedFilename, encoderType, i)
        if i == testGroup:
            testingData = encodedData
        elif trainingDataStarted:
            trainingData = np.concatenate((trainingData, encodedData))
        else:
            trainingData = encodedData
            trainingDataStarted = True
    return [trainingData, testingData]

def normaliseLabels(labels):
    minLabels = np.inf
    maxLabels = -np.inf

    for i in range(labels.shape[1]):
        for j in range(labels[0, i].shape[0]):
            label = labels[0, i][j, 0]
            if label < minLabels:
                minLabels = label
            if label > maxLabels:
                maxLabels = label
    
    # normalise the labels
    labels = (labels-minLabels)/(maxLabels-minLabels)

    return labels

def extractLabels(labels, stimuli):
    nRows = 0
    for i in stimuli:
        nRows += labels[0, i].shape[0]


    extractedLabels = np.zeros((nRows, 1))
    rowCounter = 0
    
    for i in stimuli:
        for j in range(labels[0, i].shape[0]):
                extractedLabels[rowCounter, 0] = labels[0, i][j, 0]
                rowCounter += 1

    return extractedLabels

def getLabels(filename, testGroup):
    labels = sio.loadmat('data/'+filename+'.mat')
    labels = labels[filename]

    labels = normaliseLabels(labels)

    trainingStimuli = []
    for i in range(labels.shape[1]):
        trainingStimuli += [i]

    groupSize = (int)(labels.shape[1]/10)
    testingStimuli = []
    for i in range(groupSize):
        testingStimuli += [groupSize*testGroup+i]

    trainingStimuli = [x for x in trainingStimuli if x not in testingStimuli]

    trainingLabels = extractLabels(labels, trainingStimuli)
    testingLabels = extractLabels(labels, testingStimuli)

    return [trainingLabels, testingLabels]

def getCrossedDataset(dataset):
    if dataset == 'ACC_filtered_slid_win5_lag1_LIRIS':
        return 'ACC_filtered_slid_win5_lag1_Technicolor'
    if dataset == 'ACC_filtered_slid_win5_lag1_Technicolor':
        return 'ACC_filtered_slid_win5_lag1_LIRIS'
    if dataset == 'ACC_filtered_slid_win5_lag5_LIRIS':
        return 'ACC_filtered_slid_win5_lag5_Technicolor'
    if dataset == 'ACC_filtered_slid_win5_lag5_Technicolor':
        return 'ACC_filtered_slid_win5_lag5_LIRIS'
    if dataset == 'GSR_filtered_slid_win5_lag1_LIRIS':
        return 'GSR_filtered_slid_win5_lag1_Technicolor'
    if dataset == 'GSR_filtered_slid_win5_lag1_Technicolor':
        return 'GSR_filtered_slid_win5_lag1_LIRIS'
    if dataset == 'GSR_filtered_slid_win5_lag5_LIRIS':
        return 'GSR_filtered_slid_win5_lag5_Technicolor'
    if dataset == 'GSR_filtered_slid_win5_lag5_Technicolor':
        return 'GSR_filtered_slid_win5_lag5_LIRIS'

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
