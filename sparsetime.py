"""
   sparsetime.py
  
   Dylan Paiton
   Feb 25, 2015
  
   Learn space-time basis functions
   from natural scenes

   A note on dimensionality:
     Numpy likes to specify matrices with [Z, Y, X] convention, or
     [Z, Row, Column]. Most of the adapted code uses [Y,X,T],
     but I follow numpy's [T,Y,X] convention.

"""

import numpy as np
import errno
from scipy.signal import convolve
from PIL import Image
import os
import pdb

class ImageSet:
    '''
    Time-varying images to be encoded.

    Initialize with a name and a string that contains 
    the path to a text document that lists the images
    to be analyzed, separated by new lines.
    '''

    def __init__(self,setName,imgListPath,framesPerMov):
        self.setName      = setName     #Name of dataset
        self.imgListPath  = imgListPath #Path to .txt file that contains list of images
        self.framesPerMov = -1          #Number of frames per movie
        self.loaded       = False       #
        self.whitened     = False       #
        self.dataSet      = []          #Stored image data
        self.imgCount     = -1          #Total number of images in dataset
        self.imgSizeY     = -1          #Image height
        self.imgSizeX     = -1          #Image width

    def loadImages(self):
        if not self.loaded:
            with open(self.imgListPath,'r') as fileStream:
                for line in fileStream: # each line is an absolute path to the image file
                    img = Image.open(line[0:-1]) # lines end with a '\n' char, which we don't want
                    #TODO: Don't hardcode for greyscale images
                    if img.mode is not 'L':
                        img = img.convert('L') # converts RGB to Grayscale
                    # PIL Image reports image sizes X by Y, but Numpy uses Y by X
                    imgDat = np.array(img.getdata()).reshape(img.size[1],img.size[0]).astype(np.float64)
                    self.dataSet.append(imgDat)

            self.imgCount = len(self.dataSet)
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
        ## Whitening code adapted from work by Charles Frye and Bruno Olshausen
        ## To reduce complications, cube space-time blocks is required
        ## If blockSize does not divide into self.imgCount, the loop
        ## will not whiten the final (self.imgCount%blockSize) images
        ## TODO: Whiten those last few images...
        ## TODO: Subtract mean

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

            # G = exp( -0.5 * ( [(FX.*FX + FY.*FY) / (nyqY/2)^2] + [FT.*FT / (nyqT/2)^2] ) )
            gauss = np.exp(-0.5*((np.multiply(fspace[1],fspace[1])+np.multiply(fspace[2],fspace[2]))/np.square(nyqY/2)+np.multiply(fspace[0],fspace[0])/np.square(nyqT/2)))

            powerSpec = np.zeros((blockSize,self.imgSizeY,self.imgSizeX))

            for block in range(0,self.imgCount-(self.imgCount%blockSize),blockSize):
                dataArray = np.array(self.dataSet[block:block+blockSize]) #[T,Y,X] array

                dataArray  = np.multiply(dataArray,gauss)
                blockFT    = np.fft.fftn(dataArray,axes=(0,1,2))
                pdb.set_trace()
                powerSpec += np.multiply(blockFT,np.conjugate(blockFT)) #TODO:Is this just getting the real portion?

            powerSpec = powerSpec/np.float64(len(range(0,self.imgCount-(self.imgCount%blockSize),blockSize)))

            return powerSpec


        def rotationalAverage(powSpec):
            'Compute rotational average of power spectrum (P) with dimensions [time spat freq, radial spat freq]'
            (sizeT,sizeY,sizeX) = powSpec.shape

            nyqY = np.int32(np.floor(sizeY/2))
            nyqX = np.int32(np.floor(sizeX/2))

            freqsY = np.linspace(-nyqY,nyqY-1,num=sizeY)
            freqsX = np.linspace(-nyqX,nyqX-1,num=sizeX)

            fspace = np.meshgrid(freqsY,freqsX,indexing='ij')
            rho = np.round(np.sqrt(np.square(fspace[0])+np.square(fspace[1])))
            
            indices = [None] * nyqY
            for rad in range(nyqY):
                indices[rad] = np.where(rho.ravel() == rad)[0]
            
            rotAvg = np.zeros((sizeT,nyqY))

            for tIdx in range(sizeT):
                tmp = powSpec[tIdx,:,:]
                for rad in range(nyqY):
                    rotAvg[tIdx,rad] = np.mean(tmp.ravel()[indices[rad]])

            return rotAvg


        if blockSize == -1:
            blockSize = self.imgCount

        if self.loaded:
            nyqT = np.int32(np.floor(blockSize/2))
            nyqY = np.int32(np.floor(self.imgSizeY/2))
            nyqX = np.int32(np.floor(self.imgSizeX/2))

            # Multiplier acquired from Bruno's code - not sure how it is derived
            sigT = 0.78125*nyqT
            sigY = 0.78125*nyqY
            sigX = 0.78125*nyqX

            freqsY = np.linspace(-nyqY,nyqY-1,num=self.imgSizeY)
            freqsX = np.linspace(-nyqX,nyqX-1,num=self.imgSizeX)

            # Rho is the ramp function for whitening in the spatial dimension
            fspace = np.meshgrid(freqsY,freqsX,indexing='ij')
            rho    = np.sqrt(np.square(fspace[0])+np.square(fspace[1]))
            
            # Time portion of LPF
            lowPassTime = np.exp(-(rho/sigT)**4) #TODO:Why **4?

            rho = np.int32(np.round(rho))

            filtIdx = np.where(rho.ravel() <= nyqX-1)[0] 

            filtf   = np.zeros((blockSize,self.imgSizeY,self.imgSizeX))

            powerSpec = computePowerSpectrum(blockSize)
            rotAvg    = rotationalAverage(powerSpec)

            for timeIdx in range(blockSize):
                FT = timeIdx-nyqT
                filt_time = np.zeros((self.imgSizeY,self.imgSizeX))

                # Multiply low pass time filter with spatial filter
                filt_time.ravel()[filtIdx] = np.multiply(lowPassTime.ravel()[filtIdx]*np.exp(-(FT/nyqT)**4),1./np.sqrt(rotAvg[np.abs(FT),rho.ravel()[filtIdx]]))
                filtf[timeIdx,:,:] = filt_time

            filtf[nyqT,nyqX,nyqY] = 0 # zero out the center frequency (DC value)
            filtf = np.fft.fftshift(filtf,axes=(0,1,2))

            for block in range(0,self.imgCount-(self.imgCount%blockSize),blockSize):
                dataArray    = np.array(self.dataSet[block:block+blockSize]) #[T,Y,X] array
                blockFT      = np.fft.fftshift(np.fft.fftn(dataArray,axes=(0,1,2)),axes=(0,1,2))

                # Multilpy data by filter in fourier space
                blockFT_wht  = np.multiply(blockFT,filtf)
                blockIFT_wht = np.real(np.fft.ifftn(np.fft.ifftshift(blockFT_wht,axes=(0,1,2)),axes=(0,1,2)))

                blockIFT_wht = np.sqrt(0.1)*blockIFT_wht/np.std(blockIFT_wht)

                # Turn it into an image
                if np.min(blockIFT_wht)<0:
                    blockImg_wht = blockIFT_wht + np.abs(np.min(blockIFT_wht))
                else:
                    blockImg_wht = blockIFT_wht - np.min(blockIFT_wht)

                blockImg_wht = 255 * (blockImg_wht/np.max(blockImg_wht))

                #TODO: There must be a better way to do this...
                for i in range(block,block+blockSize):
                    self.dataSet[i] = blockImg_wht[i-block,:,:]

            self.whitened = True
        else:
            print("You have to load the images before you can whiten them.")


    def getRandPatch(self,patchSizeT,patchSizeY,patchSizeX):
        # Get a random patch in space and time
        # Parameters specify the patch size, image size is property of class
        # Upper bound on rand is exclusive
        randY  = np.random.randint(0,self.imgSizeY-patchSizeY)
        randX  = np.random.randint(0,self.imgSizeX-patchSizeX)
        randT  = np.random.randint(0,self.imgCount-patchSizeT)
        patchT = np.array(self.dataSet[randT:randT+patchSizeT])
        patch  = patchT[:,randY:randY+patchSizeY,randX:randX+patchSizeX]

        return patch.reshape(patchSizeT,patchSizeY*patchSizeX)

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

    def __init__(self,sizeT,numNeurons):
        self.numNeurons     = numNeurons
        self.sizeT          = sizeT
        self.activities     = np.zeros((sizeT,numNeurons))
        self.error          = 0


    def sparseApproximation(self,imageSetArray,Dictionary,lambdaN,beta,sigma,eta,numIterations):
        '''
        Compute activities from a given dictionary & image set

        lambdaN - 
        beta    - 
        sigma   - 
        eta     - update rate for activities

        a = argmin_a [ lambda_N/2 |I(x,y,t) - sum_i {a_i * phi_i(x,y,t)}|^2 + sum_i{sum_t{S(a_i(t))}} ]
           where S(a_i(t)) = beta log(1 + (a_i(t)/sigma)^2), beta controls sparseness, sigma is scaling

        da/dt \propto lambda_N sum_x,y { corr(phi_i(x,y,t) , e(x,y,t)) - S'(a_i(t))}
           where e(x,y,t) = I(x,y,t) - sum_i{a_i(t) * phi_i(x,y,t)}

        da_i(t)/dt <- lambda_N sum_x,y{ sum_t{ phi_i(x,y,-t) (I(x,y,t) - sum_i{sum_t'{a_i(t')phi_i(t-t')}})}} - S'(a_i(t))
           where S'(a_i(t)) = (2 beta a_i(t) a'_i(t)) / (sigma^2 + a_i(t)^2)
        '''
        # First iteration
        (sizeT, sizeYX, numElements) = Dictionary.weightSet.shape
        if np.all(self.activities==0):
            # Initial activity is the time-correlation between the dictionary element & the input
            for tIdx in range(sizeT):
                self.activities += np.dot(image,Dictionary.weightSet[tIdx,:,:])

            # Normalize activities wrt weight gain coefficients
            self.activities = np.multiply(self.activities,1./np.tile(np.squre(Dictionary.gain),(sizeT,1)))

        recon = getReconstruction(Dictionary)

        self.error = imageSetArray - recon

        E = (0.5 * lambdaN * np.sum(np.square(self.error)) + beta * np.sum(S(self.activities/sigma),axis=0))/sizeT

        for iteration in range(numIterations-1):
            grada = 0
            for tIdx in range(sizeT):
                grada += lambdaN * np.dot(self.error[tIdx,:],Dictionary.weightSet[tIdx,:,:])

            grada -= (beta/sigma) * Sp(self.activities/sigma)

            da = eta*grada
            a += da

            for tIdx in range(sizeT):
                recon[tIdx,:] += np.dot(Dictionary.weightSet[tIdx,:,:]*da[tIdx,:].transpose())

            self.error = imageSetArray - recon
            Eold = E
            E = (0.5 * lambdaN * np.sum(np.square(self.error)) + beta * np.sum(S(self.activities/sigma)))/sizeT
            if (E - Eold) > 0:
                self.activities[:] = 0
                break
            
        def S(a):
            'Sparse cost function'
            sparseCost = np.log(1+np.multiply(a,a))

        def Sp(a):
            'Derivative of sparse cost function'
            sPrime = np.multiply(2*a,1./(1+np.multiply(a,a)))



    def getReconstruction(self,Dictionary,saveRecon=False,reconSavePath=''):
        (sizeT, sizeXY, numElements) = Dictionary.weightSet.shape
        recon = np.zeros((sizeT,sizeXY))
        for tIdx in range(sizeT):
            recon[tIdx,:] = np.sum(np.multiply(self.activities[tIdx,:],Dictionary.weightSet[tIdx,:,:]),axis=1)

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
        
        self.weightSet   = np.zeros((sizeT,weightPatchSize,numElements))

        #TODO: Implement other initialization modes
        if initMode[0] == 'rand':
            self.weightSet[:,:,:] = np.random.rand(sizeT,weightPatchSize,numElements)

        self.gain = np.sqrt(np.sum(np.sum(np.multiply(self.weightSet,self.weightSet),axis=1),axis=0)).transpose()

        self.initialized = True


    def computeUpdates(self,SparseNet,dPhi):
        '''
        Compute cumulative dw based on weight update rule

        Basis functions are updated by an ammount proportional to the correlation
        between the residual error e and the coefficients a.

        d phi_i(x,y,t)/dt <- < corr(a_i(t) error(x,y,t)) >

        d phi_i(x,y,t)/dt <- < a_i(t) sum_t{phi_i(x,y,-t) (I(x,y,t) - sum_i{sum_t'{a_i(t-t')phi_i(t-t')}})} >

        Inputs:
            errors = reconstruction errors computed during sparse approximation
            activities = activity values computed during sparse approximation    
            phiDim = tuple containing size of phi, which is (sizeT,patchSize,numElements) 
        '''
        if self.initialized:
            (sizeT, sizeYX, numElements) = dPhi.shape
            for timeIdx in range(sizeT):
                dPhi += SparseNet.activities.transpose() * SparseNet.getError()[::-1]

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
        normPhi = np.sqrt(np.sum(np.sum(np.multilpy(self.weightSet,self.weightSet),axis=1),axis=0))
        for elm in range(self.numElements):
            self.weightSet[:,:,elm] = gain[elm] * self.weightSet[:,:,elm] / normPhi[elm]

    #def saveWeights(self,savePath):
        # save the weights to file


    #def loadWeights(self,loadPath):
        # load the weights from file



