from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras import regularizers
import numpy as np

def getAutoencoderLayerSizes(autoencoderType, dataType):
    if autoencoderType == 0:
        if dataType == 0:
            return [161, 45]
        return [161, 35]
    if autoencoderType == 1:
        if dataType == 0:
            return [161, 103, 45]
        return [161, 98, 35]
    if dataType == 0:
        return [161, 122, 84, 45]
    return [161, 119, 77, 35]

def buildAutoencoder(layerSizes):
    inputLayer = Input(shape=(layerSizes[0],))
    layer = Dense(layerSizes[1], activation='relu', activity_regularizer=regularizers.l1(1e-7))(inputLayer)
    for layerSize in layerSizes[2:]:
        layer = Dense(layerSize, activation='relu', activity_regularizer=regularizers.l1(1e-7))(layer)
    
    encoder = Model(inputLayer, layer)

    for layerSize in reversed(layerSizes[:-1]):
        layer = Dense(layerSize, activation='sigmoid', activity_regularizer=regularizers.l1(1e-7))(layer)

    autoencoder = Model(inputLayer, layer)

    autoencoder.compile(loss='binary_crossentropy', optimizer='adadelta')

    return [autoencoder, encoder]

def buildAndTrainAutoencoder(autoencoderType, data, dataType):
    layerSizes = getAutoencoderLayerSizes(autoencoderType, dataType)

    [autoencoder, encoder] = buildAutoencoder(layerSizes)

    lastLoss = np.inf

    epochCounter = 0

    while True:
        history = autoencoder.fit(data, data, epochs=1, verbose=0)

        loss = history.history['loss'][-1]
        
        if np.linalg.norm(lastLoss - loss) < 1e-6 or epochCounter >= 240:
            break

        lastLoss = loss
        epochCounter += 1

    return [autoencoder, encoder]
 
def getRegressorLayers(regressorType, dataType):
    if regressorType == 0:
        if dataType == 0:
            return [495, 1]
        return [385, 1]
    if regressorType == 1:
        if dataType == 0:
            return [495, 248, 1]
        return [385, 193, 1]
    if dataType == 0:
        return [495, 330, 166, 1]
    return [385, 257, 129, 1]

def buildRegressor(layerSizes):
    inputLayer = Input(shape=(layerSizes[0],))
    layer = Dense(layerSizes[1], activation='relu', activity_regularizer=regularizers.l1(1e-7))(inputLayer)
    for layerSize in layerSizes[2:]:
        layer = Dense(layerSize, activation='relu', activity_regularizer=regularizers.l1(1e-7))(layer)
    
    regressor = Model(inputLayer, layer)

    return regressor

def buildAndTrainRegressor(regressorType, trainingData, targetData, dataType):
    layerSizes = getRegressorLayerSizes(regressorType, dataType)

    regressor = buildRegressor(layerSizes)

    lastLoss = np.inf

    epochCounter = 0

    while True:
        history = regressor.fit(trainingData, targetData, epochs=1, verbose=0)

        loss = history.history['loss'][-1]
        
        if np.linalg.norm(lastLoss - loss) < 1e-6 or epochCounter >= 240:
            break

        lastLoss = loss
        epochCounter += 1

        return regressor
