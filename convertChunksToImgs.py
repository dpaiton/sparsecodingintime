# From Bruno's README:
#   9600 frames, 128x128 pixels per frame, 1 byte per pixel (uchar).
#   top 14 lines of image are blank - I crop them out
#   You will notice that every pair of frames appears to be a repeat, so you can
#   average these together.  When you play at 25 f/s it looks about right.

from PIL import Image
import os
import errno

outputPath = './whitened_bruno_images/'

numImagesPerChunk = 64
imgSz             = 128

try:
    os.makedirs(outputPath)
except OSError, e:
    if e.errno != errno.EEXIST: # Only ignore error that dir exists
        raise e
    pass

chunkFiles = os.listdir('./whitened_bruno_chunks/')

numChunks = len(chunkFiles)

imgIdx = 0
for fi in chunkFiles:
    f = open('./whitened_bruno_chunks/'+fi,'r')
    for fIdx in range(numImagesPerChunk):
        #L mode is 8 bits per pixel, black & white
        img = Image.fromstring('L',(imgSz,imgSz),f.read(imgSz*imgSz))
        #img = img.crop((0,14,128,128))
        #img = img.crop((7,14,121,128))
        img.save(outputPath+str(imgIdx).zfill(4)+'.png')
        imgIdx += 1
    f.close()





