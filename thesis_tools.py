import data_tools as dt
import model_tools as mt
import numpy as np
from sklearn.metrics import mean_squared_error

def savePredictedAndEncodedData(dataset, autoencoderType):
    for i in range(10):
        [trainingData, testingData] = dt.getData(dataset, i)

        dataType = dt.getDataType(dataset)

        [autoencoder, encoder] = mt.buildAndTrainAutoencoder(autoencoderType, trainingData, dataType)

        predictedData = autoencoder.predict(testingData)

        encodedData = encoder.predict(testingData)

        np.save('model_outputs/'+dataset+'_autoencoder_'+str(autoencoderType)+'_group_'+str(i), predictedData)
        
        np.save('model_outputs/'+dataset+'_encoder_'+str(autoencoderType)+'_group_'+str(i), encodedData)

def getMetrics(dataset, autoencoderType):
    metrics = np.zeros((12, 3))
    for i in range(10):
        [_, data] = dt.getData(dataset, i)

        predictedData = np.load('model_outputs/'+dataset+'_autoencoder_'+str(autoencoderType)+'_group_'+str(i)+'.npy')

        metrics[i, 0] = mean_squared_error(data, predictedData)
        metrics[i, 1] = dt.correlationCoefficient(data, predictedData)
        metrics[i, 2] = dt.concordanceCorrelationCoefficient(data, predictedData)
    
    for i in range(3):
        metrics[10, i] = np.mean(metrics[:10, i])
        metrics[11, i] = np.std(metrics[:10, i])

    return metrics
