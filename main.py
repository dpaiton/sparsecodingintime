########
#
# An implementation of BA Olshausen (2003) -
# "Learning Sparse, Overcomplete Representations
# of Time-Varying Natural Images"
#
# Author: Dylan Paiton
# Date created: 
#
#TODO: 
# Functions for plotting recons & activity
#
#FIXME:
#
########
import sparsetime
import numpy as np

# Development libraries
import matplotlib.pyplot as plt
import skimage.io as io
import IPython

def plotDictionary(weights,margin=4):
    [sizeT, sizeYX, numElements] = weights.shape
    patchSide = np.int32(np.sqrt(sizeYX))
    elementIndices = range(numElements)
    dispWeights = np.zeros((numElements*(patchSide+margin), sizeT*(patchSide+margin)))
    dictElement = np.zeros((patchSide+margin, patchSide+margin))
    halfMargin  = np.floor(margin/2)
    xpos = 0
    ypos = 0
    for elmIdx in elementIndices:      
        for tIdx in range(sizeT): 
            dictElement[halfMargin:halfMargin+patchSide, halfMargin:halfMargin+patchSide] = \
                    weights[tIdx, :, elmIdx].reshape(patchSide, patchSide)
            dispWeights[ypos:ypos+patchSide+margin, xpos:xpos+patchSide+margin] = dictElement
            xpos += patchSide + margin
            if xpos > dispWeights.shape[1]-(patchSide+margin):
                ypos += patchSide+margin
                xpos = 0
    plt.figure()
    plt.imshow(dispWeights,cmap='Greys',interpolation='nearest')
    plt.show(block=False)

def plotActivity(activity):
    [numBatches, batchSize, sizeT, numElements] = activity.shape
    element_activity = np.zeros((numElements,numBatches*batchSize*sizeT))
    for element in range(numElements):
        element_activity[element,:] = activity[:,:,:,element].flatten(order='C')
    plt.figure()
    plt.plot(element_activity.T)
    plt.show(block=False)

# User defined paths
rootDir = "/Users/dpaiton/Work/"
imageList = rootDir+"Datasets/vanHatteren/imgList.txt"
whiteImageList = rootDir+"Datasets/vanHatteren/whiteImgList.txt"

blockSize = 64  #Number of frames per movie clip
varData = 0.1

imgSet = sparsetime.ImageSet('van Hatteren', whiteImageList, varData, blockSize)

print("Loading images...")
imgSet.loadImages()
print("Done.")

#print("Whitening images...")
#imgSet.whitenImages(blockSize)
#print("Done.")
#
#print("Writing the images to file...")
#imgWritePath = rootDir+'Datasets/vanHatteren/whitened/'
#imgSet.writeImages(imgWritePath)
#print("Done.")

#########################################################################
#### Learning ####
#
# For each batch, we load a single 114 x 114 x 64 frame sequence
#
# A 12 x 12 x 64 patch is randomly selected from the sequence
# 
# Coefficients are fitted to the equation,
# and a weight update is computed
# 
# Basis set has 200 functions, each of size 12px x 12px x 7 frames. 
#
#########################################################################

## Dictionary params
numElements = 10#200
dictSizeY   = 12
dictSizeX   = 12
dictSizeT   = 7
weightEta   = 1.0
alpha       = 0.02    # rate of adaptation for normalization (gain adjustment)
varGoal     = varData # goal variance of weights - from video
varEta      = 0.01    # adaptation rate of time constant to compute variance

## Sparse approximation params
numBatches    = 1
batchSize     = 10
noiseVar      = 0.005
beta          = 2.5   # parameters for fita - used for energy & gradient calculation
sigma         = 0.316 # fita parameters
aEta          = 0.001 # inference
numIterations = 50

## Set up additional params
aVar    = varGoal * np.ones((numElements));
lambdaN = 1. / noiseVar
margin  = np.floor(dictSizeT/2)  # Time buffer of zeros to stop edge effects with convolution.
                                 # Assumes dictSizeT is odd.
numValid = blockSize - 2*margin  # Number of valid entries in image array

print("Initializing dictionary...")
Phi  = sparsetime.Dictionary(dictSizeT,dictSizeY*dictSizeX,numElements,initMode='rand')
dPhi = np.zeros_like(Phi.weightSet) 
print("Done.")

#TODO: Initialization requrires dictionary so that arrays can be allocated to correct size
#      There is probably a better way to go about this, so that the arrays don't get allocated until
#      sparse approximation is called with a dictionary passed to it
sparseNet = sparsetime.SparseNet(Phi,blockSize) # Must be initialized with a dictionary

activityHistory = np.zeros((numBatches, batchSize, numIterations-dictSizeT, numElements))

for batch in range(numBatches):
    print("Batch number "+str(batch+1).zfill(3)+" out of "+str(numBatches).zfill(3)+"...")
    for i in range(batchSize):
        #imagePatch = sparsetime._normalize(imgSet.getRandPatch(dictSizeY, dictSizeX))
        imagePatch = imgSet.getRandPatch(dictSizeY, dictSizeX)
        updateRateMod = 1 
        [energy, error] = sparseNet.sparseApproximation(imagePatch, Phi, lambdaN, beta, sigma, aEta/updateRateMod, numIterations)
        dPhi = Phi.computeUpdates(sparseNet, dPhi)
        recon = sparseNet.getReconstruction(Phi, sparseNet.activities, True)
        #plotDictionary(dPhi)
        activityHistory[batch, i, :, :] = sparseNet.activities
        #IPython.embed()

    aVar = (1-varEta) * aVar + varEta * \
        np.sum(np.multiply(sparseNet.activities, sparseNet.activities), axis=0) / numValid

    dPhi /= batchSize * numValid 
    Phi.updateWeights(dPhi,weightEta)
    Phi.normalizeWeights(aVar,varGoal,sigma,alpha)
    sparseNet.gain = np.sqrt(np.sum(np.sum(np.multiply(Phi.weightSet,Phi.weightSet),axis=1),axis=0)).transpose()

plotActivity(activityHistory)

plt.figure()
plt.plot(np.mean(sparseNet.error,axis=1))
plt.show(block=False)

IPython.embed()
