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
# Development libraries
import numpy as np

# Local libraries
import sparsetime
import plot_functions as pf

# Debugging libraries
import IPython
import matplotlib.pyplot as plt


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
#########################################################################

## Dictionary params
numElements = 200
dictSizeY   = 12
dictSizeX   = 12
dictSizeT   = 7
weightEta   = 1.0
alpha       = 0.02    # rate of adaptation for normalization (gain adjustment)
varGoal     = varData # goal variance of weights - from video
varEta      = 0.01    # adaptation rate of time constant to compute variance

## Sparse approximation params
numBatches    = 500
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
print("Done.")

#TODO: Initialization requrires dictionary so that arrays can be allocated to correct size
#      There is probably a better way to go about this, so that the arrays don't get allocated until
#      sparse approximation is called with a dictionary passed to it
sparseNet = sparsetime.SparseNet(Phi,blockSize) # Must be initialized with a dictionary

recFigNo  = None
dPhiFigNo = None

(phiFigNo, phiFigAxes) = pf.showWeightsFromImg(Phi.makeWeightImg(numSubImages=8))
(varFigNo, varFigAxes) = pf.plotBars(data=(Phi.gain, aVar), titles=('Phi', 'aVar'))
(actFigNo, actImgObj) = pf.showActivity(sparseNet.activities)

imagePatch = sparsetime._normalize(imgSet.getRandPatch(dictSizeY, dictSizeX))
(recFigNo, subAxes) = pf.showRecons(imagePatch, imagePatch, sparseNet.error)

IPython.embed()

for batch in range(numBatches):
    print("\nBatch number "+str(batch+1).zfill(3)+" out of "+str(numBatches).zfill(3)+"...\n")

    dPhi = np.zeros_like(Phi.weightSet) 
    for i in range(batchSize):
        print("Trial number "+str(i+1).zfill(3)+" out of "+str(batchSize).zfill(3)+"...")

        sparseNet.clearActivities()
        while np.all(sparseNet.activities == 0): # Get new patch if current patch caused instability
            imagePatch = sparsetime._normalize(imgSet.getRandPatch(dictSizeY, dictSizeX))
            #imagePatch = imgSet.getRandPatch(dictSizeY, dictSizeX) # If I don't normalize the patch, my energy blows up

            updateRateMod = 1 
            while np.all(sparseNet.activities == 0) and updateRateMod <= 4: # Divide eta by 2 if network is unstable, only make 3 attempts
                [energy, error] = sparseNet.sparseApproximation(imagePatch, Phi, lambdaN, beta, sigma, aEta/updateRateMod, numIterations)
                updateRateMod *= 2

            #IPython.embed()

        dPhi = Phi.computeUpdates(sparseNet, dPhi)
        aVar = (1-varEta) * aVar + \
                varEta * np.sum(np.multiply(sparseNet.activities, sparseNet.activities), axis=0) / numValid

        (recFigNo, subAxes) = pf.showRecons(imagePatch, sparseNet.getReconstruction(Phi), error[-1,:,:], (recFigNo, subAxes))
        #dPhiFigNo = pf.showDictionary(dPhi, dPhiFigNo)
        #IPython.embed()

    dPhi /= batchSize * numValid 
    Phi.updateWeights(dPhi,weightEta)
    Phi.normalizeWeights(aVar,varGoal,sigma,alpha)
    sparseNet.gain = np.sqrt(np.sum(np.sum(np.multiply(Phi.weightSet,Phi.weightSet),axis=1),axis=0)).transpose()

    (phiFigNo, phiFigAxes) = pf.showWeightsFromImg(Phi.makeWeightImg(numSubImages=8), (phiFigNo, phiFigAxes))
    #(varFigNo, varFigAxes) = pf.plotBars(data=(Phi.gain, aVar), titles=('Phi', 'aVar'), prevFig=(varFigNo, varFigAxes))


print "Done"
IPython.embed()
