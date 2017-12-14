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

def getAutoencoderLoss(layerSizes, trainingData, validationData):
    [autoencoder, _] = buildAutoencoder(layerSizes)
    history = autoencoder.fit(trainingData, trainingData, epochs=30, verbose=0, validation_data=(validationData, validationData))

    return history.history['loss'][-1]

def findAutoencoderArchitecture(trainingData, validationData):
    layerSizes = [trainingData.shape[1]]

    withoutNewLayerLoss = np.inf

    l = (np.sqrt(5)-1)/2

    simplicityBias = 1e-4

    while True:
        print('Searching for optimal architecture with '+str(len(layerSizes)*2-1)+' hidden layers\n')
        # find the optimal size for a new layer
        s0 = 1
        s3 = layerSizes[-1]
        s1 = (int)(np.ceil(l*s0+(1-l)*s3))
        l1 = getAutoencoderLoss(layerSizes+[s1], trainingData, validationData)
        s2 = (int)(np.floor((1-l)*s0+l*s3))
        l2 = getAutoencoderLoss(layerSizes+[s2], trainingData, validationData)

        while s2>s0 and s1<s3:
            print('Searching between '+str(s0)+' and '+str(s3)+' units for new hiddenmost layer...\n')
            if l1>l2+simplicityBias:
                s0 = s1
                s1 = s2
                l1 = l2
                s2 = (int)(np.floor((1-l)*s0+l*s3))
                l2 = getAutoencoderLoss(layerSizes+[s2], trainingData, validationData)
            else:
                s3 = s2
                s2 = s1
                l2 = l1
                s1 = (int)(np.ceil(l*s0+(1-l)*s3))
                l1 = getAutoencoderLoss(layerSizes+[s1], trainingData, validationData)

        print('Optimal new hiddenmost layer found with '+str(s1)+' units\n')

        if withoutNewLayerLoss>l1+simplicityBias:
            print('New hiddenmost layer incorporated\n')
            layerSizes += [s1]
            withoutNewLayerLoss = l1
        else:
            print('New hiddenmost layer discarded\n')
            print('Optimal architecture found : \n')
            for ls in layerSizes:
                print(ls)
            for ls in reversed(layerSizes[:-1]):
                print(ls)
                print('\n')
            break
            print('Loss : ' + str(withoutNewLayerLoss) + '\n')

    return layerSizes

def buildAndTrainAutoencoder(layerSizes, trainingData, validationData):
    [autoencoder, encoder] = buildAutoencoder(layerSizes)

    autoencoder.fit(trainingData, trainingData, epochs=60, verbose=0, validation_data=(validationData, validationData))

    return [autoencoder, encoder]

def saveAutoencoder(autoencoder, encoder, name):
    autoencoder.save('saved_models/'+name+'_autoencoder.h5')
    encoder.save('saved_models/'+name+'_encoder.h5')

def loadAutoencoder(name):
    autoencoder = load_model('saved_models/'+name+'_autoencoder.h5')
    encoder = load_model('saved_models/'+name+'_encoder.h5')
    
    return [autoencoder, encoder]
