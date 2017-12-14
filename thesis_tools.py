import data_tools as dt
import autoencoder_tools as at
import time
import numpy as np

def findAndSaveModel(filename):
    print('Retrieving data...\n')
    startTimer = time.perf_counter()
    startProcessorTimer = time.process_time()

    data = dt.getNormalisedData(filename)

    [trainingData, validationData] = dt.splitData(data)

    print('Data retrieved :')
    print('Time : ' + str(time.perf_counter()-startTimer))
    print('Processor time : ' + str(time.process_time()-startProcessorTimer) + '\n')

    print('Finding the optimal architecture for the data...\n')
    startTimer = time.perf_counter()
    startProcessorTimer = time.process_time()

    layerSizes = at.findAutoencoderArchitecture(trainingData, validationData)

    print('Optimal architecture found :')
    print('Time : ' + str(time.perf_counter()-startTimer))
    print('Processor time : ' + str(time.process_time()-startProcessorTimer) + '\n')

    print('Training the model...\n')
    startTimer = time.perf_counter()
    startProcessorTimer = time.process_time()

    [autoencoder, encoder] = at.buildAndTrainAutoencoder(layerSizes, trainingData, validationData)

    print('Model trained :')
    print('Time : ' + str(time.perf_counter()-startTimer))
    print('Processor time : ' + str(time.process_time()-startProcessorTimer) + '\n')

    at.saveAutoencoder(autoencoder, encoder, filename)

    test = autoencoder.predict(validationData)

    avgDist = 0

    for i in range(test.shape[0]):
        avgDist += np.linalg.norm(test[i, :]-validationData[i, :])

    avgDist /= test.shape[0]
    
    print(avgDist)
