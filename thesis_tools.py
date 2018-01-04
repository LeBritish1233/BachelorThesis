import data_tools as dt
import autoencoder_tools as at
import numpy as np

def savePredictedAndEncodedData(dataset, autoencoderType):
    for i in range(10):
        [trainingData, testingData] = dt.getData(dataset, i)

        dataType = dt.getDataType(dataset)

        [autoencoder, encoder] = at.buildAndTrainAutoencoder(autoencoderType, trainingData, dataType)

        predictedData = autoencoder.predict(testingData)

        encodedData = encoder.predict(testingData)

        np.save('model_outputs/'+dataset+'_autoencoder_'+str(autoencoderType)+'_group_'+str(i), predictedData)
        
        np.save('model_outputs/'+dataset+'_encoder_'+str(autoencoderType)+'_group_'+str(i), encodedData)
