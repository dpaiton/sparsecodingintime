import matplotlib.pyplot as plt
import numpy as np

import IPython

def plotActivityHistory(activity_history):
    [numBatches, batchSize, sizeT, numElements] = activity_history.shape
    element_activity = np.zeros((numElements,numBatches*batchSize*sizeT))
    for element in range(numElements):
        element_activity[element,:] = activity_history[:,:,:,element].flatten(order='C')
    figNo = plt.figure()
    plt.plot(element_activity.T)
    figNo.show()
    return figNo

def plotBars(data, titles, prevFig=None):
    assert len(data) == len(titles)
    if prevFig is None:
        figNo, figSubAxes = plt.subplots(len(data),1)
    else:
        (figNo, subAxes) = prevFig
    for idx, datum in enumerate(data):
        figSubAxes[idx].bar(range(0, 2*len(datum), 2), datum)                                  # adding space between bars
        figSubAxes[idx].set_xticks(range(0, 2*len(datum)+1, 100))                              # print labels as if space was not there
        figSubAxes[idx].set_xticklabels([str(x).zfill(2) for x in range(0, len(datum)+1, 50)]) # set labels to normal range
        figSubAxes[idx].set_title(titles[idx])
    if prevFig is None:
        figNo.show()
    else:
        figNo.canvas.draw()
    return (figNo, figSubAxes)

def showActivity(activity, prevFig=None):
    if prevFig is None:
        figNo = plt.figure()
        actImg = plt.imshow(activity.T, cmap='Greys', vmin=-1, vmax=1)
        plt.axis('image')
        figNo.show()
    else:
        (figNo, actImg) = prevFig
        actImg.set_data(activity.T)
        figNo.canvas.draw()
    return (figNo, actImg)

def showRecons(img, recon, error, prevFig=None):
    [sizeT, sizeYX] = img.shape
    imgSide = np.int32(np.floor(np.sqrt(sizeYX)))
    if prevFig is None:
        figNo, subAxes = plt.subplots(3, 1)
    else:
        (figNo, subAxes) = prevFig
    subAxes[0].imshow(img.reshape(imgSide, imgSide*sizeT), cmap='Greys', vmin=-1, vmax=1)
    subAxes[1].imshow(recon.reshape(imgSide, imgSide*sizeT), cmap='Greys', vmin=-1, vmax=1)
    subAxes[2].imshow(error.reshape(imgSide, imgSide*sizeT), cmap='Greys', vmin=-1, vmax=1)
    for axis in figNo.axes:
        axis.get_yaxis().set_visible(False)
        axis.get_xaxis().set_visible(False)
    if prevFig is None:
        figNo.show()
    else:
        figNo.canvas.draw()
    return (figNo, subAxes)

def showWeights(weights, margin=4):
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
    if prevFig is None:
        figNo = plt.figure()
        weightImg = plt.imshow(dispWeights, cmap='Greys', interpolation='nearest')
        figNo.show()
    else:
        (figNo, weightImg) = prevFig
        weightImg.set_data(dispWeights)
        figNo.canvas.draw()
    return (figNo, weightImg)

def showWeightsFromImg(weightImg, prevFig=None):
        [numSubPlots, elementSz, timeSz] = weightImg.shape
        if prevFig is None:
            figNo, subAxes = plt.subplots(1,numSubPlots)
        else:
            (figNo, subAxes) = prevFig
        for subPlotIdx, subAxis in enumerate(subAxes):
            subAxis.imshow(weightImg[subPlotIdx,:,:], cmap='Greys', interpolation='nearest')
        for axis in figNo.axes:
            axis.get_yaxis().set_visible(False)
            axis.get_xaxis().set_visible(False)
        if prevFig is None:
            figNo.show()
        else:
            figNo.canvas.draw()

        return (figNo, subAxes)
