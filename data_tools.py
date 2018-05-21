import scipy.io as sio
import numpy as np

"""

getDataType lets us know whether the data contained within a file corresponds to Acceleration or Galvanic Skin Response

@param filename : a String corresponding to the name of a file

@returns 0 if Acceleration or 1 if Galvanic Skin Response

"""

def getDataType(filename):
    if filename[0:3] == "ACC":
        return 0
    return 1

"""

normaliseData normalises the data extracted from a file

@param data : a table of movies containing tables of spectators containing matrices of data

@returns a normalised version of data

"""

def normaliseData(data):
    # find the minimum and maximum value within the data
    # initilise the necessary variables
    minData = np.inf
    maxData = -np.inf

    # explore the data
    for i in range(data.shape[1]):
        for j in range(data[0, 0].shape[1]):
            for k in range(data[0, i][0, 0].shape[0]):
                # select a row and find its minimum and maximum
                row = data[0, i][0, j][k, :]
                minRow = min(row)
                maxRow = max(row)
                # compare the row's minimum/maximum with the data's current minimum/maximum, replace if lower/greater
                if minRow < minData:
                    minData = minRow
                if maxRow > maxData:
                    maxData = maxRow
    
    # normalise the data
    data = (data-minData)/(maxData-minData)
    
    return data

"""

extractData extracts the data for desired movies from given data

@param data : a table of movies containing tables of spectators containing matrices of data
@param stimuli : the desired stimuli ie the movies for which we want data

@returns the extracted data reshaped into a matrix

"""

def extractData(data, stimuli):
    # find the size of the desired normalised data
    nRows = 0
    nCols = data[0, 0][0, 0].shape[1]
    for i in stimuli:
        for j in range(data[0, 0].shape[1]):
            nRows += data[0, i][0, j].shape[0]


    # copy the data into the normalised data matrix
    extractedData = np.zeros((nRows, nCols))
    rowCounter = 0
    
    for i in stimuli:
        for j in range(data[0, 0].shape[1]):
            for k in range(data[0, i][0, 0].shape[0]):
                # add the row to the normalised data matrix
                extractedData[rowCounter, :] = data[0, i][0, j][k, :]
                rowCounter += 1

    return extractedData

"""

getData extracts the training and testing data from a file

@param filename : a String corresponding to the name of a file
@param testGroup : an integer going from 0 to 9 which indicates the movies selected for testing data

@returns the training and testing data in matrix form

"""

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

"""

getEncodedData gets the data that has been encoded by the encoder part of the autoencoder for a group of movies and extracts the data for which we have labels

@param filename : a String which corresponds the the name of a file
@param encoderType : an indication of the type of encoder used to encode the data
@param group : an integer going from 0 to 9 which indicates the group of movies which is of interest

@returns the encoded data for which we have labels

"""

def getEncodedData(filename, encoderType, group):
    # get the data
    data = sio.loadmat('data/'+filename+'.mat')
    data = data[filename]
 
    # get the size of a group of stimuli ie movies
    groupSize = (int)(data.shape[1]/10)

    # create a table containing the group stimuli
    groupStimuli = np.zeros(groupSize, int)
    for i in range(groupSize):
        groupStimuli[i] = groupSize*group+i

    # get the sizes of the data for the desired stimuli
    groupStimuliSizes = np.zeros(groupSize, int)
    for i in range(groupSize):
        groupStimuliSizes[i] = data[0, groupStimuli[i]][0, 0].shape[0]

    # get the labels ie emotions for the data
    if 'lag1' in filename:
        labels = sio.loadmat('data/Arousal_all_movies_win5_lag1.mat')
        labels = labels['Arousal_all_movies_win5_lag1']
    else:
        labels = sio.loadmat('data/Arousal_all_movies_win5_lag5.mat')
        labels = labels['Arousal_all_movies_win5_lag5']

    # create a table indicating the amount of data for which we have labels
    desiredGroupStimuliSizes = np.zeros(groupSize, int)
    for i in range(groupSize):
        desiredGroupStimuliSizes[i] = labels[0, groupStimuli[i]].shape[0]

    # get the encoded data
    encodedData = np.load('model_outputs/'+filename+'_encoder_'+str(encoderType)+'_group_'+str(group)+'.npy')

    # extract the data for which we have labels from the encoded data
    reshapedEncodedData = np.zeros((sum(desiredGroupStimuliSizes), data[0, 0].shape[1]*encodedData.shape[1]))

    rc = 0

    for i in range(groupSize):
        for j in range(data[0, 0].shape[1]):
            for k in range(desiredGroupStimuliSizes[i]):
                reshapedEncodedData[k+sum(desiredGroupStimuliSizes[:i]), j*encodedData.shape[1]:(j+1)*encodedData.shape[1]] += encodedData[rc, :]
                rc += 1
                if k == desiredGroupStimuliSizes[i]-1:
                    rc += groupStimuliSizes[i]-desiredGroupStimuliSizes[i]

    return reshapedEncodedData

"""

getEncodedDataCrossed works like getEncodedData, but for data on which used cross-database training

@param filename : a String which corresponds to the name of the file for which we encoded data
@param crossedFilename : a String which corresponds to the name of the file used for training
@param encoderType : an integer going from 0 to 2 indicating the encoder used
@param group : an integer going from 0 to 9 indicating the group

@returns the encoded data for which we have labels

"""

def getEncodedDataCrossed(filename, crossedFilename, encoderType, group):
    # the code functions very much like the code for the getEncodedData function
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

"""

getRegressionInputData gets the data that will be used as input for the regression model

@param filename : a String which corresponds to the name of the file for which we encoded data
@param encoderType : an integer going from 0 to 2 indicating the encoder used
@param testGroup : an integer going from 0 to 9 indicating the group used for testing

@returns the training and testing data for the regression model

"""

def getRegressionInputData(filename, encoderType, testGroup):
    # we initialise a boolean value to know if we have the commenced building the training data
    trainingDataStarted = False
    for i in range(10):
        # we go through the groups and get the encoded data
        encodedData = getEncodedData(filename, encoderType, i)
        # if the group is the test group then we have testing data
        if i == testGroup:
            testingData = encodedData
        # otherwise we add the data to the training data if we are building it
        elif trainingDataStarted:
            trainingData = np.concatenate((trainingData, encodedData))
        # otherwise we start building training data
        else:
            trainingData = encodedData
            trainingDataStarted = True
    return [trainingData, testingData]

"""

getRegressionInputDataCrossed works like getRegressionInputData, but for data on which used cross-database training

@param filename : a String which corresponds to the name of the file for which we encoded data
@param crossedFilename : a String which corresponds to the name of the file used for training
@param encoderType : an integer going from 0 to 2 indicating the encoder used
@param testGroup : an integer going from 0 to 9 indicating the group used for testing

@returns the training and testing data for the regression model

"""
def getRegressionInputDataCrossed(filename, crossedFilename, encoderType, testGroup):
    # the code is almost exactly like that of getRegressionInputData
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

"""

normaliseLabels normalises the labels

@param labels : a matrix containing the labels

returns the normalised labels

"""

def normaliseLabels(labels):
    # find the min and the max of labels
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

"""

extractLabels extracts the labels corresponding to given stimuli

@param labels : a matrix containing the labels
@param stimuli : a list of integers which indicate the stimuli

@returns the desired labels

"""

def extractLabels(labels, stimuli):
    # initilise the necessary variables
    nRows = 0
    for i in stimuli:
        nRows += labels[0, i].shape[0]


    extractedLabels = np.zeros((nRows, 1))
    rowCounter = 0
    
    # get the labels for the stimuli
    for i in stimuli:
        for j in range(labels[0, i].shape[0]):
                extractedLabels[rowCounter, 0] = labels[0, i][j, 0]
                rowCounter += 1

    return extractedLabels

"""

getLabels gets the labels for training and testing data

@param filename : a String corresponding to the name of the file being used for getting data
@param testGroup : an integer going drom 0 to 9 indicating which group is used for testing

returns the labels for the training and testing data

"""

def getLabels(filename, testGroup):
    # get the normalised labels
    labels = sio.loadmat('data/'+filename+'.mat')
    labels = labels[filename]

    labels = normaliseLabels(labels)
    
    # get the training stimuli
    trainingStimuli = []
    for i in range(labels.shape[1]):
        trainingStimuli += [i]

    # get the testing stimuli
    groupSize = (int)(labels.shape[1]/10)
    testingStimuli = []
    for i in range(groupSize):
        testingStimuli += [groupSize*testGroup+i]

    # redefine the training stimuli for 
    trainingStimuli = [x for x in trainingStimuli if x not in testingStimuli]

    # get the labels for the data
    trainingLabels = extractLabels(labels, trainingStimuli)
    testingLabels = extractLabels(labels, testingStimuli)

    return [trainingLabels, testingLabels]

"""

getCrossedDataset gives the dataset used for training when using the cross-dataset training strategy

@param dataset : a String corresponding to the used dataset

@returns the name of the dataset used for training

"""

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

"""

correlationCoefficient gets the correlation coefficient for two sets of data

@param data : the actual data
@param predictedData : the data predicted by models

returns the correlation coefficient

"""

def correlationCoefficient(data, predictedData):
    n = 0
    d1 = 0
    d2 = 0
    md = np.mean(data)
    mpd = np.mean(predictedData)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            n += np.absolute((data[i, j]-md))*np.absolute((predictedData[i, j]-mpd))
            d1 += np.absolute((data[i, j]-md))*np.absolute((data[i, j]-md))
            d2 += np.absolute((predictedData[i, j]-mpd))*np.absolute((predictedData[i, j]-mpd))

    return n/np.sqrt(d1*d2)

"""

concordanceCorrelationCoefficient gets the correlation coefficient for two sets of data

@param data : the actual data
@param predictedData : the data predicted by models

returns the concordance correlation coefficient

"""

def concordanceCorrelationCoefficient(data, predictedData):
    cc = correlationCoefficient(data, predictedData)
    md = np.mean(data)
    mpd = np.mean(predictedData)
    stdd = np.std(data)
    stdpd = np.std(predictedData)

    return 2*cc*stdd*stdpd/(stdd*stdd+stdpd*stdpd+(md-mpd)*(md-mpd))
