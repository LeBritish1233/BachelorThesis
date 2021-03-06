import data_tools as dt
import model_tools as mt
import numpy as np
from sklearn.metrics import mean_squared_error

"""

savePredictedEncodedData saves the reconstructed and encoded data

@param dataset : a String corresponding tho the name of the datafile
@param autoencoderType : an integer going from 0 to 2

"""

def savePredictedAndEncodedData(dataset, autoencoderType):
    for i in range(10):
        # for every group we get the training and testing data
        [trainingData, testingData] = dt.getData(dataset, i)

        dataType = dt.getDataType(dataset)

        # we build the autoencoder and train it
        [autoencoder, encoder] = mt.buildAndTrainAutoencoder(autoencoderType, trainingData, dataType)

        # we save the data
        predictedData = autoencoder.predict(testingData)

        encodedData = encoder.predict(testingData)

        np.save('model_outputs/'+dataset+'_autoencoder_'+str(autoencoderType)+'_group_'+str(i), predictedData)
        
        np.save('model_outputs/'+dataset+'_encoder_'+str(autoencoderType)+'_group_'+str(i), encodedData)

"""

getMetricsPredictedData gets the different errors

@param dataset : a String corresponding to the name of the dataset
@param autoencoderType : a number going from 0 to 2

@returns the different metrics

"""

def getMetricsPredictedData(dataset, autoencoderType):
    metrics = np.zeros((12, 3))
    # get metrics for the different groups
    for i in range(10):
        [_, data] = dt.getData(dataset, i)

        predictedData = np.load('model_outputs/'+dataset+'_autoencoder_'+str(autoencoderType)+'_group_'+str(i)+'.npy')

        metrics[i, 0] = mean_squared_error(data, predictedData)
        metrics[i, 1] = dt.correlationCoefficient(data, predictedData)
        metrics[i, 2] = dt.concordanceCorrelationCoefficient(data, predictedData)
    
    # get the average for each metrics
    for i in range(3):
        elements = []
        for j in range(10):
            if metrics[j, i] == metrics[j, i]:
                elements += [metrics[j, i]]
        metrics[10, i] = np.mean(elements)
        metrics[11, i] = np.std(elements)

    return metrics

"""

savePredictedEncodedDataCrossed saves the reconstructed and encoded data for the crossed training strategy

@param dataset : a String corresponding tho the name of the datafile
@param autoencoderType : an integer going from 0 to 2

"""

def savePredictedAndEncodedDataCrossed(dataset, autoencoderType):
    # this code works like savePredictedEncodedData
    crossedDataset = dt.getCrossedDataset(dataset)

    for i in range(10):
        [trainingData, _] = dt.getData(crossedDataset, i)

        [_, testingData] = dt.getData(dataset, i)

        dataType = dt.getDataType(dataset)

        [autoencoder, encoder] = mt.buildAndTrainAutoencoder(autoencoderType, trainingData, dataType)

        predictedData = autoencoder.predict(testingData)

        encodedData = encoder.predict(testingData)

        np.save('model_outputs/'+dataset+'_crossed_'+crossedDataset+'_autoencoder_'+str(autoencoderType)+'_group_'+str(i), predictedData)

        np.save('model_outputs/'+dataset+'_crossed_'+crossedDataset+'_encoder_'+str(autoencoderType)+'_group_'+str(i), encodedData)

"""

getMetricsPredicedDataCrossed gets the errors for data using cross-database strategy

@param dataset : a String corresponding tho the name of the datafile
@param autoencoderType : an integer going from 0 to 2

@returns the metrics

"""

def getMetricsPredictedDataCrossed(dataset, autoencoderType):
    # this function works like getMetricsPredictedData
    metrics = np.zeros((12, 3))
    crossedDataset = dt.getCrossedDataset(dataset)
    for i in range(10):
        [_, data] = dt.getData(dataset, i)

        predictedData = np.load('model_outputs/'+dataset+'_crossed_'+crossedDataset+'_autoencoder_'+str(autoencoderType)+'_group_'+str(i)+'.npy')

        metrics[i, 0] = mean_squared_error(data, predictedData)
        metrics[i, 1] = dt.correlationCoefficient(data, predictedData)
        metrics[i, 2] = dt.concordanceCorrelationCoefficient(data, predictedData)
    
    for i in range(3):
        elements = []
        for j in range(10):
            if metrics[j, i] == metrics[j, i]:
                elements += [metrics[j, i]]
        metrics[10, i] = np.mean(elements)
        metrics[11, i] = np.std(elements)

    return metrics

"""

saveRegressedData saves the regressed data

@param dataset : a String corresponding to the name of the dataset
@param labelset : a String corresponding to the name of the labelset
@param encodedType : an integer going from 0 to 2

"""

def saveRegressedData(dataset, labelset, encoderType):
    # this code works like that of the autoencoder
    for i in range(10):
        [trainingData, testingData] = dt.getRegressionInputData(dataset, encoderType, i)

        [trainingLabels, _] = dt.getLabels(labelset, i)

        dataType = dt.getDataType(dataset)

        regressor = mt.buildAndTrainRegressor(trainingData, trainingLabels, dataType)

        predictedLabels = regressor.predict(testingData)

        np.save('model_outputs/'+dataset+'_'+labelset+'_encoder_'+str(encoderType)+'_group_'+str(i), predictedLabels)

"""

getMetricsRegressedData gets the metrics for the regressedData

@param dataset : a String corresponding to the name of the dataset
@param labelset : a String coreesponding to the name of the labelset
@param an integer going from 0 to 2

@returns the metrics

"""

def getMetricsRegressedData(dataset, labelset, encoderType):
    # this code will work like that of the autoencoder
    metrics = np.zeros((12, 3))
    for i in range(10):
        [_, labels] = dt.getLabels(labelset, i)

        predictedLabels = np.load('model_outputs/'+dataset+'_'+labelset+'_encoder_'+str(encoderType)+'_group_'+str(i)+'.npy')

        metrics[i, 0] = mean_squared_error(labels, predictedLabels)
        metrics[i, 1] = dt.correlationCoefficient(labels, predictedLabels)
        metrics[i, 2] = dt.concordanceCorrelationCoefficient(labels, predictedLabels)

    for i in range(3):
        elements = []
        for j in range(10):
            if metrics[j, i] == metrics[j, i]:
                elements += [metrics[j, i]]
        metrics[10, i] = np.mean(elements)
        metrics[11, i] = np.std(elements)

    return metrics

"""

saveMetricsRegressedDataCrossed gets the metrics for the regressedData

@param dataset : a String corresponding to the name of the dataset
@param labelset : a String coreesponding to the name of the labelset
@param an integer going from 0 to 2

"""

def saveRegressedDataCrossed(dataset, labelset, encoderType):
    # this code works like that of the autoencoder
    crossedDataset = dt.getCrossedDataset(dataset)
    for i in range(10):
        [trainingData, testingData] = dt.getRegressionInputDataCrossed(dataset, crossedDataset, encoderType, i)

        [trainingLabels, _] = dt.getLabels(labelset, i)

        dataType = dt.getDataType(dataset)

        regressor = mt.buildAndTrainRegressor(trainingData, trainingLabels, dataType)

        predictedLabels = regressor.predict(testingData)

        np.save('model_outputs/'+dataset+'_crossed_'+crossedDataset+'_'+labelset+'_encoder_'+str(encoderType)+'_group_'+str(i), predictedLabels)

"""

getMetricsRegressedDataCrossed gets the metrics for the regressedData

@param dataset : a String corresponding to the name of the dataset
@param labelset : a String coreesponding to the name of the labelset
@param an integer going from 0 to 2

@returns the metrics

"""

def getMetricsRegressedDataCrossed(dataset, labelset, encoderType):
    # this code works like that of the autoencoder
    metrics = np.zeros((12, 3))
    crossedDataset = dt.getCrossedDataset(dataset)
    for i in range(10):
        [_, labels] = dt.getLabels(labelset, i)

        predictedLabels = np.load('model_outputs/'+dataset+'_crossed_'+crossedDataset+'_'+labelset+'_encoder_'+str(encoderType)+'_group_'+str(i)+'.npy')

        metrics[i, 0] = mean_squared_error(labels, predictedLabels)
        metrics[i, 1] = dt.correlationCoefficient(labels, predictedLabels)
        metrics[i, 2] = dt.concordanceCorrelationCoefficient(labels, predictedLabels)

    for i in range(3):
        elements = []
        for j in range(10):
            if metrics[j, i] == metrics[j, i]:
                elements += [metrics[j, i]]
        metrics[10, i] = np.mean(elements)
        metrics[11, i] = np.std(elements)

    return metrics
