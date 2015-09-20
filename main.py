import sparsetime
import numpy as np
import pdb

#imageList = '/Users/dpaiton/Documents/Datasets/vanHatteren/imgList.txt'
imageList = '/Users/dpaiton/Documents/Datasets/vanHatteren/whiteImgList.txt'

blockSize = 64  #Number of frames per movie clip

imgSet = sparsetime.ImageSet('van Hatteren',imageList,blockSize)

print("Loading images...")
imgSet.loadImages()
print("Done.")

#print("Whitening images...")
#imgSet.whitenImages(blockSize)
#print("Done.")
#
#print("Writing the images to file...")
#imgWritePath = '/Users/dpaiton/Documents/Datasets/vanHatteren/whitened/'
#imgSet.writeImages(imgWritePath)
#print("Done.")

# For each batch, we load a single 114 x 114 x 64 frame sequence
#
# A 12 x 12 x 64 patch is randomly selected from the sequence
# 
# Coefficients are fitted to the equation, and a weight update is computed
# 
# Basis set has 200 functions, each of size 12px x 12px x 7 frames. 
#

## Dictionary params
numElements = 200
dictSizeY  = 12
dictSizeX  = 12
dictSizeT  = 7
weight_eta  = 1.0
alpha       = 0.02
varGoal     = 0.1
var_eta     = 0.01

## Sparse approximation params
batchSize     = 10
noise_var     = 0.005
beta          = 2.5
sigma         = 0.316
a_eta         = 0.001
numIterations = 50

## Set up additional params
aVar       = varGoal*np.ones((numElements));
lambdaN    = 1./noise_var
margin       = np.floor(dictSizeT/2)    # Time buffer of zeros to stop edge effects with convolution.
                                        # Assumes dictSizeT is odd.
numValid     = blockSize - 2*margin     # Number of valid entries in image array

Phi  = sparsetime.Dictionary(dictSizeT,dictSizeY*dictSizeX,numElements,initMode='rand')
dPhi = np.zeros_like(Phi.weightSet) 

#TODO: Initialization requrires dictionary so that arrays can be allocated to correct size
#      There is probably a better way to go about this, so that the arrays don't get allocated until
#      sparse approximation is called with a dictionary passed to it
sparseNet = sparsetime.SparseNet(Phi,blockSize) # Must be initialized with a dictionary


for i in range(batchSize):
    print("Batch number "+str(i+1).zfill(2)+" out of "+str(batchSize).zfill(2)+"...")
    while sparseNet.activities.all == 0:
        imagePatch = imgSet.getRandPatch(dictSizeY,dictSizeX)
        updateRateMod = 1 
        while sparseNet.activities.all == 0:
            sparseNet.sparseApproximation(imagePatch,Phi,lambdaN,beta,sigma,a_eta/updateRateMod,numIterations)
            updateRateMod *= 2
    dPhi = Phi.computeUpdates(sparseNet,dPhi)
    aVar = (1-var_eta) * aVar + var_eta * np.sum(np.multiply(sparseNet.activities,sparseNet.activities),axis=0)/numValid
    Phi.plotWeights(margin=4,numElements=10,indices=[0,1,2,3,4,5,6,7,8,9])
dPhi /= batchSize * numValid 
Phi.updateWeights(dPhi,weight_eta)
Phi.normalizeWeights(aVar,varGoal,sigma,alpha)
