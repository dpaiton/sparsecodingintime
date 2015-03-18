import sparsetime
import numpy as np

#imageList = '/Users/dpaiton/Documents/Datasets/vanHatteren/imgList.txt'
imageList = '/Users/dpaiton/Documents/Datasets/vanHatteren/whiteImgList.txt'

blockSize = 64  #Number of frames per movie clip

imgSet = sparsetime.ImageSet('van Hatteren',imageList,blockSize)

print("Loading images...")
imgSet.loadImages()
print("Done.")

print("Whitening images...")
imgSet.whitenImages(blockSize)
print("Done.")

print("Writing the images to file...")
imgWritePath = '/Users/dpaiton/Documents/Datasets/vanHatteren/whitened/'
imgSet.writeImages(imgWritePath)
print("Done.")

# For each batch, we load a single 114 x 114 x 64 frame sequence
#
# A random 12 x 12 x 7 sub patch is randomly selected from the sequence
# 
# Coefficients are fitted to the equation, and a weight update is computed
# 
# Basis set has 200 functions, each of size 12px x 12px x 7 frames. 
#

### Dictionary params
#numElements = 200
#patchSizeY  = 12
#patchSizeX  = 12
#patchSizeT  = 7
#numNonZero  = 0.25 # Percent initialized to non-zero values
#weight_eta  = 1.0
#alpha       = 0.02
#varGoal     = 0.1
#var_eta     = 0.01
#
### Sparse approximation params
#noise_var     = 0.005
#beta          = 2.5
#sigma         = 0.316
#a_eta         = 0.001
#numIterations = 50
#
### Set up additional params
#aVar      = varGoal*np.ones((numElements,1));
#lambdaN   = 1./noise_var
#
#Phi = sparsetime.Dictionary(patchSizeT,patchSizeY*patchSizeX,numElements,initMode=('rand',numNonZero))
#dPhi      = 0
#for i in range(batchSize):
#    imagePatch = imgSet.getRandPatch(patchSizeT,patchSizeY,patchSizeX)
#    SparseNet.sparseApproximation(imagePatch,Phi,lambdaN,beta,sigma,a_eta,numIterations)
#    dPhi = Dictionary.computeUpdates(SparseNet,dPhi)
#    aVar = (1-var_eta) * aVar + var_eta * np.sum(np.dot(SparseNet.activities,np.square(SparseNet.activities)),axis=1) / sizeT
#dPhi /= batchSize * sizeT
#Dictionary.updateWeights(dPhi,eta)
#Dictionary.normalizeWeights(aVar,varGoal,sigma,alpha)
