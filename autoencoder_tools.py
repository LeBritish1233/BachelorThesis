from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras import regularizers
import numpy as np

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

def buildAndTrainAutoencoder(hiddenLayerSize, data, targetData, autoencoderLearningConvergence, autoencoderMaxEpochs):
    layerSizes = [data.shape[1], hiddenLayerSize]

    [autoencoder, encoder] = buildAutoencoder(layerSizes)

    lastLoss = np.inf

    epochCounter = 0

    while True:
        history = autoencoder.fit(data, targetData, epochs=1, verbose=0, validation_split = 0.1)

        loss = history.history['loss'][-1]
      
        print(np.linalg.norm(loss - lastLoss))
        if np.linalg.norm(loss - lastLoss) < autoencoderLearningConvergence or epochCounter >= autoencoderMaxEpochs:
            break

        lastLoss = loss
        epochCounter += 1

    return [autoencoder, encoder]

def saveAutoencoder(autoencoder, encoder, name):
    autoencoder.save('saved_models/'+name+'_autoencoder.h5')
    encoder.save('saved_models/'+name+'_encoder.h5')

def loadAutoencoder(name):
    autoencoder = load_model('saved_models/'+name+'_autoencoder.h5')
    encoder = load_model('saved_models/'+name+'_encoder.h5')
    
    return [autoencoder, encoder]
