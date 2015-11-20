"""
   sparsetime.py
  
   Dylan Paiton
   Feb 25, 2015
  
   Learn space-time basis functions
   from natural scenes

   Assumptions:
     * All frames are grayscale
     * All video clips in dataset are the same length

   Matrices follow the Numpy [T(time),Y(row),X(column)] convention.

"""

import os
import errno

from scipy.signal import convolve
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from PIL import Image

import IPython

def _normalize(mat):
    if np.min(mat) == np.max(mat):
        return mat
    else:
        return (mat - np.min(mat)) / (np.max(mat) - np.min(mat))
        #normVal = np.max([np.abs(np.max(mat)), np.abs(np.min(mat))])
        #return mat / normVal

class ImageSet:
    '''
    Time-varying images to be encoded.

    Initialize with a name and a string that contains 
    the path to a text document that lists the images
    to be analyzed, separated by new lines.
    '''

    def __init__(self, setName, imgListPath, varData, framesPerMov=-1):
        self.setName      = setName      # Name of dataset
        self.imgListPath  = imgListPath  # Path to .txt file that contains list of images
        self.framesPerMov = framesPerMov # Number of frames per movie
        self.varData      = varData      # Variance of data after whitening
        self.loaded       = False        # 
        self.whitened     = False        # 
        self.dataSet      = []           # Stored image data
        self.imgCount     = -1           # Total number of images in dataset
        self.imgSizeY     = -1           # Image height
        self.imgSizeX     = -1           # Image width

    def loadImages(self):
        if not self.loaded:
            with open(self.imgListPath,'r') as fileStream:
                for line in fileStream: # each line is an absolute path to the image file
                    #TODO: Switch this to skimage instead of PIL
                    img = Image.open(line[0:-1]) # lines end with a '\n' char, which we don't want
                    #TODO: Don't hardcode for greyscale images
                    if img.mode is not 'L':
                        img = img.convert('L') # converts RGB to Grayscale
                    # PIL Image reports image sizes X by Y, but Numpy uses Y by X
                    imgDat = np.array(img.getdata()).reshape(img.size[1],img.size[0]).astype(np.float64)
                    self.dataSet.append(imgDat)

            self.imgCount = len(self.dataSet)

            if self.framesPerMov == -1:
                self.framesPerMov = self.imgCount

            (self.imgSizeY,self.imgSizeX) = self.dataSet[0].shape

            self.loaded = True
        else:
            print("Already loaded in the image set for ",self.setName)
    
    def displayStats(self):
        if self.loaded:
            print("Number of images: {0}\nImage height: {1}\nImage width: {2}".format(self.imgCount,self.imgSizeY,self.imgSizeX))
        else:
            print("Load images before requesting stats.")

    def whitenImages(self, blockSize=-1):
        '''
         To reduce complications, cube space-time blocks is required
         If blockSize does not divide into self.imgCount, the loop
         will not whiten the final (self.imgCount%blockSize) images
         TODO: Whiten those last few images...
         TODO: Subtract mean
         TODO: This code does spatial whitening * time LPF.
               I need to write an alternative function that
               performs joint whitening over space & time.
        '''

        def computePowerSpectrum(blockSize):

            (sizeY, sizeX) = self.dataSet[0].shape

            nyqT = np.int32(np.floor(blockSize/2))
            nyqY = np.int32(np.floor(sizeY/2))
            nyqX = np.int32(np.floor(sizeX/2))

            freqsT = np.linspace(-nyqT,nyqT-1,num=blockSize)
            freqsY = np.linspace(-nyqY,nyqY-1,num=self.imgSizeY)
            freqsX = np.linspace(-nyqX,nyqX-1,num=self.imgSizeX)

            # Generate a gaussian window function
            fspace = np.meshgrid(freqsT,freqsY,freqsX,indexing='ij')

            # G = exp( -0.5 * ( (FT/(nyqT/2))^2 + (FY/(nyqY/2))^2 + (FX/(nyqX/2))^2 ) )
            gauss = np.exp(-0.5 * (np.square(fspace[0]/(nyqT/2)) +
                np.square(fspace[1]/(nyqY/2)) +
                np.square(fspace[2]/(nyqX/2))))

            powerSpec = np.zeros((blockSize,self.imgSizeY,self.imgSizeX))

            trange = range(0,self.imgCount-(self.imgCount%blockSize),blockSize)

            for block in trange:
                dataArray = np.array(self.dataSet[block:block+blockSize]) #[T,Y,X] array
                dataArray = (dataArray - np.min(dataArray)) / (np.max(dataArray) - np.min(dataArray))
                dataArray = dataArray * gauss
                blockFT   = np.fft.fftn(dataArray,axes=(0,1,2))
                powerSpec = np.multiply(blockFT,np.conjugate(blockFT)).real

            powerSpec = powerSpec/np.float64(len(trange))

            return powerSpec

        def rotationalAverage(powSpec):
            '''
            Compute rotational average of power spectrum (P) with dimensions [time spat freq, radial spat freq]
            Only works for greyscale imagery
            '''

            dims = powSpec.shape

            nyq = np.int32(np.floor(np.array(dims)/2.0))

            freqs = [np.linspace(-nyq[i],nyq[i]-1,num=dims[i]) for i in range(len(dims))]

            if len(dims) == 3: # time domain included, and expected to be first dim
                fspace = np.meshgrid(freqs[1], freqs[2],indexing='ij')

                # Rho is cartesian coordinate for (x,y) in Fourier space
                rho = np.round(np.sqrt(np.square(fspace[0])+np.square(fspace[1])))
                
                rotAvg = np.zeros((dims[0],rho.shape[0]))
                for time in range(dims[0]):
                    tmp = powSpec[time,:,:]
                    for rad in range(rho.shape[0]):
                        if np.isnan(np.mean(tmp[rho == rad])):
                            rotAvg[time,rad] = 0
                        else:
                            rotAvg[time,rad] = np.mean(tmp[rho == rad])
            else:
                fspace = np.meshgrid(freqs[0],freqs[1],indexing='ij')
                rho = np.sqrt(np.square(fspace[0])+np.square(fspace[1]))

                rotAvg = np.zeros(rho.shape[0])
                for rad in range(rho.shape[0]):
                    if np.isnan(np.mean(powSpec[rho == rad])):
                        rotAvg[rad] = 0
                    else:
                        rotAvg[rad] = np.mean(powSpec[rho == rad])

            return rotAvg

        if blockSize == -1:
            blockSize = self.imgCount

        if self.loaded:
            white_filt = np.zeros((blockSize, self.imgSizeY, self.imgSizeX))

            nyq = np.array((np.int32(np.floor(self.imgSizeY/2.0)),
                   np.int32(np.floor(self.imgSizeX/2.0))))

            # Ramp function to whiten
            grid = np.mgrid[-nyq[0]:nyq[0]+(self.imgSizeY%(nyq[0]*2)), -nyq[1]:nyq[1]+(self.imgSizeX%(nyq[1]*2))]
            ramp_filter = _normalize(np.sqrt(np.square(grid[0]) + np.square(grid[1])))
            
            # spatial Gaussian Low Pass Filter
            sig = 0.78125*nyq # Cutoff frequencies
            spatial_LPF = np.exp(-0.5*(np.square(ramp_filter/(sig[0]/2.0)) + np.square(ramp_filter/(sig[1]/2.0))))

            nyq_t = np.int32(np.floor(blockSize/2.0))
            sig_t = 0.78125 * nyq_t

            for time in range(blockSize):
                FT = time - nyq_t
                time_LPF = np.exp(-0.5*np.square(-FT/(sig_t/2.0)))
                white_filt[time,:,:] = ramp_filter * spatial_LPF * time_LPF 

            white_filt = np.fft.fftshift(white_filt,axes=(0,1,2))
            white_filt[nyq_t,nyq[0],nyq[1]] = 0 # zero out the center frequency (DC value)

            for block in range(0,self.imgCount-(self.imgCount%blockSize),blockSize):
                dataArray = _normalize(np.array(self.dataSet[block:block+blockSize])) #[T,Y,X] array
                blockFT   = np.fft.fftn(dataArray,axes=(0,1,2))

                # Multilpy data by filter in fourier space
                blockIFT_wht = np.fft.ifftn(white_filt * blockFT, axes=(0,1,2)).real

                # Set variance of data to self.varData
                blockIFT_wht = np.sqrt(self.varData)*blockIFT_wht/np.std(blockIFT_wht)

                # Reformat for writing
                blockImg_wht = 255 * _normalize(blockIFT_wht)
                
                #TODO: There must be a better way to do this...
                for i in range(block,block+blockSize):
                    self.dataSet[i] = blockImg_wht[i-block,:,:]

            self.whitened = True
        else:
            print("You have to load the images before you can whiten them.")


    def getRandPatch(self,patchSizeY,patchSizeX):
        # Get a random patch in space, but contiguous in time 
        # If dataset contains multiple movies (i.e. framesPerMov != imgCount), it will pick a random movie
        # Parameters specify the patch size, image size is property of class
        # It will always make the movie self.framesPerMov long
        # Upper bound on rand is exclusive

        if self.loaded:
            numMovies = self.imgCount / self.framesPerMov # Assumes all movies are the same length

            randY   = np.random.randint(0,self.imgSizeY-patchSizeY)
            randX   = np.random.randint(0,self.imgSizeX-patchSizeX)

            randMov = np.random.randint(0,numMovies)

            startFrame = randMov * self.framesPerMov

            patchT  = np.array(self.dataSet[startFrame:startFrame+self.framesPerMov])
            patch   = patchT[:,randY:randY+patchSizeY,randX:randX+patchSizeX]

            return patch.reshape(self.framesPerMov,patchSizeY*patchSizeX)
        else:
            print("You have to load the images before you can get a patch.")
            return -1

    def writeImages(self,imgWritePath,imgIndices=-1):
        if imgIndices == -1:
            imgIndices = range(self.imgCount)
        try:
            os.makedirs(imgWritePath)
        except OSError, e:
            if e.errno != errno.EEXIST: # Only ignore error that dir exists
                raise e
            pass

        numDigits = np.int32(np.floor(np.log10(np.max(imgIndices)+1)+1)) # for padding

        for imgIdx,imgArry in enumerate(self.dataSet):
            img = Image.fromarray(imgArry).convert('L')
            img.save(imgWritePath+str(imgIdx).zfill(numDigits)+'.png')


class SparseNet:
    '''
    Symbolic network of neurons for sparse approximation
    
    Presumes square dictionary elements as input
    '''

    def __init__(self,Dictionary,setSizeT):
        (dictSizeT, sizeYX, numElements) = Dictionary.weightSet.shape
        self.numNeurons = numElements 
        self.setSizeT   = setSizeT
        # activities are initialized to the size of the image set minus a margin on either end to resolve convolution edge effects
        self.activities = np.zeros((setSizeT-dictSizeT,numElements)) 
        self.error      = np.zeros((setSizeT,sizeYX))


    def sparseApproximation(self,imageSetArray,Dictionary,lambdaN,beta,sigma,eta,numIterations):
        '''
        Compute activities from a given dictionary & image set

        lambdaN - 
        beta    - sparseness
        sigma   - scaling
        eta     - update rate for activities

        E = argmin_a [ lambda_N/2 |I(x,y,t) - sum_i {a_i * phi_i(x,y,t)}|^2 + sum_i{sum_t{S(a_i(t))}} ]
           where S(a_i(t)) = beta log(1 + (a_i(t)/sigma)^2), beta controls sparseness, sigma is scaling

        da/dt \propto lambda_N sum_x,y { corr(phi_i(x,y,t) , e(x,y,t)) - S'(a_i(t))}
           where e(x,y,t) = I(x,y,t) - sum_i{a_i(t) * phi_i(x,y,t)}

        da_i(t)/dt <- lambda_N sum_x,y{ sum_t{ phi_i(x,y,-t) (I(x,y,t) - sum_i{sum_t'{a_i(t')phi_i(t-t')}})}} - S'(a_i(t))
           where S'(a_i(t)) = (2 beta a_i(t) a'_i(t)) / (sigma^2 + a_i(t)^2)
        '''

        def S(a):
            'Sparse cost function'
            return np.log(1 + np.multiply(a, a))

        def Sp(a):
            'Derivative of sparse cost function'
            return np.multiply(2 * a, 1. / (1 + np.multiply(a, a)))

        (dictSizeT, sizeYX, numElements) = Dictionary.weightSet.shape

        energy       = np.zeros((numIterations))
        error        = np.zeros((numIterations,self.setSizeT,sizeYX))
        margin       = np.floor(dictSizeT/2) # Time buffer of zeros to stop edge effects with convolution. Assumes dictSizeT is odd.
        numValid     = self.setSizeT - 2 * margin # number of valid entries in image array
        startBuffIdx = np.arange(0,margin,dtype=np.int)
        endBuffIdx   = np.arange(self.setSizeT-margin,self.setSizeT,dtype=np.int)

        imageSetArray[startBuffIdx,:] = 0
        imageSetArray[endBuffIdx,:]   = 0

        # Set activity to dot product on first iteration
        if np.all(self.activities==0): # Check to make sure we are not using a pre-initialized activity set
            # Initial activity is the time-correlation between the dictionary element & the input
            for tIdx in range(dictSizeT):
                imgSetWindow = np.arange(tIdx, self.setSizeT-(dictSizeT-tIdx), dtype=int) # Sliding window to compute activity
                # Sum to get average response for dictionary at across the whole input scene
                self.activities = np.dot(imageSetArray[imgSetWindow,:], Dictionary.weightSet[tIdx,:,:])
        
            # Normalize activities wrt weight gain coefficients
            self.activities = np.multiply(self.activities,
                    1. / np.tile(np.square(Dictionary.gain), (self.activities.shape[0], 1)))

        recon = self.getReconstruction(Dictionary)

        error[0,:,:] = imageSetArray - recon
        error[0,startBuffIdx,:] = 0
        error[0,endBuffIdx,:]   = 0

        energy[0] = (1.5 * lambdaN * np.sum(np.square(error[0,:,:])) + beta * np.sum(S(self.activities/sigma)))/numValid

        # The rest of the iterations
        for iteration in range(1,numIterations):
            grada = 0
            for tIdx in range(dictSizeT):
                imgSetWindow = np.arange(tIdx,self.setSizeT-(dictSizeT-tIdx),dtype=int) # Sliding window to compute activity
                grada += lambdaN * np.dot(error[iteration,imgSetWindow,:], Dictionary.weightSet[tIdx,:,:])

            grada -= (beta/sigma) * Sp(self.activities/sigma)

            self.activities += eta * grada 

            recon = self.getReconstruction(Dictionary)

            error[iteration,:,:] = imageSetArray - recon
            error[iteration,startBuffIdx,:] = 0
            error[iteration,endBuffIdx,:]   = 0

            energy[iteration] = (0.5 * lambdaN * np.sum(np.square(error[iteration,:,:])) + beta * np.sum(S(self.activities/sigma)))/numValid
            #if energy[iteration] > energy[iteration-1]:
            #    print "Warning: Energy not decreasing on iteration "+str(iteration).zfill(2)+"."
            #    #self.activities[:] = 0
            #    #break

        self.error = error[numIterations-1,:,:]
        return energy, error
            

    def plotReconstruction(self, Dictionary, activity=-1):
        recon = getReconstruction(Dictionary, activity

    def getReconstruction(self, Dictionary, activity=-1, plotRecon=False, saveRecon=False, reconSavePath=''):
        if type(activity) is not np.ndarray: # If activity has not been given as input, use member variable
            activity = self.activities

        (dictSizeT, sizeYX, numElements) = Dictionary.weightSet.shape

        #TODO: Normalization is not in Bruno's code, but without it I'm not sure how it is guaranteed to be between 0 & 1
        recon = np.zeros((self.setSizeT,sizeYX))
        for tIdx in range(dictSizeT):
            imgSetWindow = np.arange(tIdx,self.setSizeT-(dictSizeT-tIdx),dtype=int) # Sliding window to compute activity
            recon[imgSetWindow,:] += _normalize(np.dot(activity,Dictionary.weightSet[tIdx,:,:].transpose()))

        recon /= len(range(dictSizeT))

        if plotRecon:
            IPython.embed()

        #if saveRecon:
            #TODO: Save recons to reconSavePath 
            
        return recon


class Dictionary:
    'Learned basis set'

    def __init__(self,weightSizeT,weightPatchSize,numElements,initMode='rand'):
        '''
        Initialize weight matrix

        Inputs:
            weightSizeT = Size of matrix in number of time frames
            weightPatchSize = Total size of weight patch. Patch dimensions will be np.sqrt(weightPatchSize) per side.
            initMode = tuple representing (mode, [additionalParams]).
                       mode is a string, and additional params depends on mode.
                       
                       Currently, the only supported mode is 'rand' and additional param is numNonZero - the number
                       of elements initialized to random non-zero values within (0,1].

        '''
        self.sizeT       = weightSizeT
        self.patchSize   = weightPatchSize
        self.numElements = numElements
        
        self.weightSet   = np.zeros((weightSizeT,weightPatchSize,numElements))

        if type(initMode) is str:
            initType = initMode
        elif type(initMode) is tuple:
            initType = initMode[0]

        #TODO: Implement other initialization modes
        if initType == 'rand':
            self.weightSet[:,:,:] = np.random.rand(weightSizeT,weightPatchSize,numElements)

        self.gain = np.sqrt(np.sum(np.sum(np.multiply(self.weightSet,self.weightSet),axis=1),axis=0)).transpose()

        self.initialized = True


    def computeUpdates(self,sparseNet,dPhi):
        '''
        Compute cumulative dw based on weight update rule

        Basis functions are updated by an ammount proportional to the correlation
        between the residual error e and the coefficients a.

        d phi_i(x,y,t)/dt \propto < corr(a_i(t) error(x,y,t)) >

        d phi_i(x,y,t)/dt \propto < a_i(t) sum_t{phi_i(x,y,-t) (I(x,y,t) - sum_i{sum_t'{a_i(t-t')phi_i(t-t')}})} >

        Inputs:
            errors = reconstruction errors computed during sparse approximation
            activities = activity values computed during sparse approximation    
            phiDim = tuple containing size of phi, which is (sizeT,patchSize,numElements) 
        '''
        if self.initialized:
            (dictSizeT, sizeYX, numElements) = dPhi.shape
            for tIdx in range(dictSizeT):
                imgSetWindow = np.arange(tIdx,sparseNet.setSizeT-(dictSizeT-tIdx),dtype=int) # Sliding window to compute activity
                dPhi[tIdx,:,:] += np.dot(sparseNet.error[imgSetWindow,:].transpose(),sparseNet.activities)

            return dPhi
        else:
            print('Need to initialize weights before you can compute updates.')
            return -1


    def updateWeights(self,dPhi,eta):
        # actually do the update
        self.weightSet += eta*dPhi


    def normalizeWeights(self,aVar,varGoal,sigma,alpha):
        '''
        Without normalization, weights will grow without bound.

        Rescale basis functions so that their L2 norm, g_i = | phi_i |_L2, 
        maintains an appropriate level of variance on each corresponding
        coefficient a_i:

        g_i^new = g_i^old [ <a_i^2> / sigma^2 ]^alpha

        sigma should be the same as what is used in sparseApproximation
        alpha is the rate of adaptation

        '''
        self.gain = np.multiply(self.gain,(aVar/varGoal)**alpha)
        normPhi = np.sqrt(np.sum(np.sum(np.multiply(self.weightSet,self.weightSet),axis=1),axis=0))
        for elm in range(self.numElements):
            self.weightSet[:,:,elm] = self.gain[elm] * self.weightSet[:,:,elm] / normPhi[elm]


    def plotWeights(self,margin=4,numElements=-1,indices='all'):
        '''
        Plots weights

        Assumes that patches are square
        '''
        patchSide = np.sqrt(self.patchSize)

        if numElements == -1:
            numElements = self.numElements
            indices = 'all'

        if indices is 'rand':
            elementIndices = np.random.random_integers(0,self.numElements-1,numElements)
        elif indices is 'all':
            elementIndices = range(numElements)
        else:
            elementIndices = indices

        # Out matrix is backwards from normal convention (elements-by-time instead of the other way around).
        # This is so that it is formatted for matplotlib.
        dispWeights = np.zeros((numElements*(patchSide+margin),self.sizeT*(patchSide+margin)))
        
        dictElement = np.zeros((patchSide+margin,patchSide+margin))
        halfMargin  = np.floor(margin/2)
        xpos = 0
        ypos = 0
        for elmIdx in elementIndices: # element traverses rows 
            for tIdx in range(self.sizeT):     # t traverses columns 
                dictElement[halfMargin:halfMargin+patchSide,halfMargin:halfMargin+patchSide] = self.weightSet[tIdx,:,elmIdx].reshape(patchSide,patchSide)
                dispWeights[ypos:ypos+patchSide+margin,xpos:xpos+patchSide+margin] = dictElement

                xpos += patchSide+margin
                if xpos > dispWeights.shape[1]-(patchSide+margin):
                    ypos += patchSide+margin
                    xpos = 0

        plt.figure()
        plt.imshow(dispWeights,cmap='Greys',interpolation='nearest')
        plt.show(block=False)

    #def saveWeights(self,savePath):
        # save the weights to file

    #def loadWeights(self,loadPath):
        # load the weights from file
