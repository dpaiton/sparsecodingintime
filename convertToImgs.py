# From Bruno's README:
#   9600 frames, 128x128 pixels per frame, 1 byte per pixel (uchar).
#   top 14 lines of image are blank - I crop them out
#   You will notice that every pair of frames appears to be a repeat, so you can
#   average these together.  When you play at 25 f/s it looks about right.

from PIL import Image
import os
import errno

outputPath = './Images/'
numImages  = 9600
imgSz      = 128

try:
    os.makedirs(outputPath)
except OSError, e:
    if e.errno != errno.EEXIST: # Only ignore error that dir exists
        raise e
    pass

f = open('vid075','r')
for fIdx in range(numImages):
    #L mode is 8 bits per pixel, black & white
    if fIdx%2 == 0: #average with every other frame
        prevImg = Image.fromstring('L',(imgSz,imgSz),f.read(imgSz*imgSz))
        #crop wants a 4-tuple with (left,upper,right,lower)
        #prevImg = prevImg.crop((0,14,128,128))
        prevImg = prevImg.crop((7,14,121,128))
    else:
        img = Image.fromstring('L',(imgSz,imgSz),f.read(imgSz*imgSz))
        #img = img.crop((0,14,128,128))
        img = img.crop((7,14,121,128))
        img = Image.blend(prevImg,img,0.5) #combine prev frame with current frame
        img.save(outputPath+str((fIdx-1)/2).zfill(4)+'.png')
f.close()





