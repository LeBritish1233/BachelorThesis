from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras import regularizers
import numpy as np

"""

getAutoencoderLayerSizes gives the sizes for the different layers of the model

@param autoencoderType : an integer going from 0 to 2
@param dataType : an integer going from 0 to 1

@returns a list of integers containing the layer sizes

"""

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

"""

buildAutoencoder builds an autoencoder

@param layerSizes : a list of integers corresponding the sizes of layers

@returns the autoencoder and the encoder part

"""

def buildAutoencoder(layerSizes):
    # build the encoder
    inputLayer = Input(shape=(layerSizes[0],))
    layer = Dense(layerSizes[1], activation='relu', activity_regularizer=regularizers.l1(1e-7))(inputLayer)
    for layerSize in layerSizes[2:]:
        layer = Dense(layerSize, activation='relu', activity_regularizer=regularizers.l1(1e-7))(layer)
    
    encoder = Model(inputLayer, layer)

    # build the decoder
    for layerSize in reversed(layerSizes[:-1]):
        layer = Dense(layerSize, activation='sigmoid', activity_regularizer=regularizers.l1(1e-7))(layer)

    autoencoder = Model(inputLayer, layer)

    autoencoder.compile(loss='binary_crossentropy', optimizer='adadelta')

    return [autoencoder, encoder]

"""

buildAndTrainAutoencoder builds and trains an autoencoder

@param autoencoderType : an integer going from 0 to 2
@param data : the data
@param dataType : an integer going from 0 to 1

@returns the trained autoencoder and encoder

"""

def buildAndTrainAutoencoder(autoencoderType, data, dataType):
    # build the autoencoder
    layerSizes = getAutoencoderLayerSizes(autoencoderType, dataType)

    [autoencoder, encoder] = buildAutoencoder(layerSizes)

    # train the autoencoder until the loss has converged or it has gone through a sufficient number of epochs
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

"""

getRegressorLayerSizes gives the layer sizes for the regressor

@param dataType : an integer going from 0 to 1

@returns the layer sizes

"""
 
def getRegressorLayerSizes(dataType):
    if dataType == 0:
        return [495, 330, 166, 1]
    return [385, 257, 128 ,1]

"""

buildRegressor builds the regressor

@param layerSizes : a list of integers containing the sizes of the layers

@returns the regressor

"""

def buildRegressor(layerSizes):
    inputLayer = Input(shape=(layerSizes[0],))
    layer = Dense(layerSizes[1], activation='relu', activity_regularizer=regularizers.l1(1e-7))(inputLayer)
    for layerSize in layerSizes[2:]:
        layer = Dense(layerSize, activation='relu', activity_regularizer=regularizers.l1(1e-7))(layer)
    
    regressor = Model(inputLayer, layer)
    
    regressor.compile(loss='binary_crossentropy', optimizer='adadelta')

    return regressor

"""

buildAndTrainRegressor builds and trains the regressor

@param trainingData : the data used for training
@param targetData : the target data
@dataType : an integer going from 0 to 1

@returns the trained regressor

"""

def buildAndTrainRegressor(trainingData, targetData, dataType):
    # the code for this function is similar to that of the autoencoder
    layerSizes = getRegressorLayerSizes(dataType)

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
