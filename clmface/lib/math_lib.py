import math
import numpy as np
from copy import deepcopy

def procrustes(data1, data2):
    assert len(data1) == len(data2)
    template = deepcopy(data1)
    shape = deepcopy(data2)
    templateMean = np.mean(template, axis=0)
    shapeMean = np.mean(shape, axis=0)
    template -= templateMean
    shape -= shapeMean
    scaleS = 0
    scaleT = 0
    for i in range(len(shape)):
        scaleS += (shape[i][0]*shape[i][0] + shape[i][1]*shape[i][1])
        scaleT += (template[i][0]*template[i][0] + template[i][1]*template[i][1])
    scaling = math.sqrt(scaleT/1.0/scaleS)
    shape *= scaling
    top = 0
    bottom = 0
    for i in range(len(shape)):
        top += shape[i][0]*template[i][1]-shape[i][1]*template[i][0]
        bottom += shape[i][0]*template[i][0]+shape[i][1]*template[i][1]

    rotation = math.atan(top/bottom)
    translateX = templateMean[0]-scaling*(math.cos(rotation)*shapeMean[0]+math.sin(-rotation)*shapeMean[1]);
    translateY = templateMean[1]-scaling*(math.sin(rotation)*shapeMean[0]+math.cos(rotation)*shapeMean[1]);
    return [translateX, translateY, scaling, rotation]

# depreciated, slower than np.vectorize
def logistic(response):
    h, w = response.shape
    for i in range(h):
        for j in range(w):
            response[i][j]=1.0/(1.0+math.exp(-(response[i][j]-1)))

# depreciated, way slower than simple numpy arithmetic
def normalize(response):
    maxv = -0.00001
    minv = 10000000
    h, w = response.shape
    for i in range(h):
        for j in range(w):
            if response[i][j] > maxv:
                maxv = response[i][j]
            if response[i][j] < minv:
                minv = response[i][j]
    dist = maxv - minv
    if dist != 0:
        for i in range(h):
            for j in range(w):
                response[i][j] = (response[i][j] - minv)/1.0/dist
