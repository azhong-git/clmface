import os
import cv2
import numpy as np
import math
from math_lib import procrustes

# np.round is different than IEEE round
# instead of using np.round, vectorize round below
np_round = np.vectorize(round)

def getInitialPosition(img, model, harrModelDir, debug=False):
    # first get an initial guess of where the face is
    face_cascade = cv2.CascadeClassifier(os.path.join(harrModelDir, 'frontal_face.xml'))
    eye_cascade = cv2.CascadeClassifier(os.path.join(harrModelDir, 'eye.xml'))
    nose_cascade = cv2.CascadeClassifier(os.path.join(harrModelDir, 'nose.xml'))
    faces = face_cascade.detectMultiScale(img)
    detected = False
    # has to find one face to start with
    if len(faces) == 1:
        (fx, fy, fw, fh) = faces[0]
        face_crop = img[fy:fy+fh,fx:fx+fw]
        eyes = eye_cascade.detectMultiScale(face_crop)
        noses = nose_cascade.detectMultiScale(face_crop, minNeighbors=6)
        if len(eyes) == 2 and len(noses) == 1:
            if eyes[0][0] < eyes[1][0]:
                left_eye_rect = eyes[0]
                right_eye_rect = eyes[1]
            else:
                left_eye_rect = eyes[1]
                right_eye_rect = eyes[0]
            left_eye = [left_eye_rect[0] + left_eye_rect[2]/2.0 + fx, left_eye_rect[1] + left_eye_rect[3]/2.0 + fy]
            right_eye = [right_eye_rect[0] + right_eye_rect[2]/2.0 + fx, right_eye_rect[1] + right_eye_rect[3]/2.0 + fy]
            nose = [noses[0][0] + noses[0][2]/2.0 + fx, noses[0][1] + noses[0][3]/2.0 + fy]

            if debug:
                import matplotlib.pyplot as plt
                dst = img.copy()
                cv2.rectangle(dst, (fx, fy), (fx+fw, fy+fh), (255),2)
                cv2.circle(dst, (int(left_eye[0]), int(left_eye[1])), 5, (255))
                cv2.circle(dst, (int(right_eye[0]), int(right_eye[1])), 5, (255))
                cv2.circle(dst, (int(nose[0]), int(nose[1])), 5, (255))
                plt.imshow(dst, cmap='gray')
                plt.title('intial frontal face guess')
                plt.show()
            procrustes_params = procrustes([left_eye, right_eye, nose],
                                           [model['hints']['leftEye'], model['hints']['rightEye'], model['hints']['nose']])
            return True, procrustes_params
    return False, (None, None, None, None)

# depreciated, use calculatePositions2
def calculatePositions(parameters, meanShape, eigenVectors, useTransforms):
    numParameters = len(parameters)-4
    numPatches = len(meanShape)
    positions = []
    for i in range(0, numPatches):
        x = meanShape[i][0]
        y = meanShape[i][1]
        for j in range(0, numParameters):
            x+=(eigenVectors[i*2][j]*parameters[j+4])
            y+=(eigenVectors[i*2+1][j]*parameters[j+4])
        if useTransforms:
            # AZ: tuning
            a = parameters[0]*x - parameters[1]*y + parameters[2];
            b = parameters[0]*y + parameters[1]*x + parameters[3];
            x += a;
            y += b;
        positions.append([x,y])
    return positions

# faster version of calculatePositions
def calculatePositions2(parameters, meanXShape, meanYShape, xEigenVectors, yEigenVectors, useTransforms):
    numPatches = len(meanXShape)
    x = meanXShape + np.dot(xEigenVectors, parameters[4:]).reshape(numPatches, 1)
    y = meanYShape + np.dot(yEigenVectors, parameters[4:]).reshape(numPatches, 1)
    xy = np.hstack((x, y))
    if useTransforms:
        homogeneous = np.ones((numPatches, 1))
        xy1 = np.hstack((xy, homogeneous))
        XY = np.dot(xy1, [[parameters[0]+1, parameters[1]],
                          [-parameters[1],   parameters[0]+1],
                          [parameters[2],   parameters[3]]])
        return np_round(xy).astype(np.int), XY
    else:
        return np_round(xy).astype(np.int), None

def convertPatchVectorToPositionVector(parameters, dPatch):
    dx, dy = dPatch
    dX = (parameters[0]+1)*dx - parameters[1]*dy
    dY = (parameters[0]+1)*dy + parameters[1]*dx
    return (dX, dY)

# depreciated, use createJacobian2
def createJacobian(parameters, meanShape, eigenVectors):
    numParameters = len(parameters)-4
    numPatches = len(meanShape)
    jacobian = np.zeros((2*numPatches, numParameters+4))
    for i in range(numPatches):
        # 1
        j0 = meanShape[i][0]
        j1 = meanShape[i][1]
        for p in range(numParameters):
            j0+=parameters[p+4]*eigenVectors[i*2][p]
            j1+=parameters[p+4]*eigenVectors[i*2+1][p]
        jacobian[i][0] = j0
        jacobian[i+numPatches][0] = j1
        # 2
        j0 = meanShape[i][1]
        j1 = meanShape[i][0]
        for p in range(numParameters):
            j0 += parameters[p+4]*eigenVectors[i*2+1][p];
            j1 += parameters[p+4]*eigenVectors[i*2][p];
        jacobian[i][1] = -j0;
        jacobian[i+numPatches][1] = j1;
        # 3
        jacobian[i][2] = 1;
        jacobian[i+numPatches][2] = 0;
        # 4
        jacobian[i][3] = 0;
        jacobian[i+numPatches][3] = 1;
        # the rest
        for j in range(numParameters):
            j0 = parameters[0]*eigenVectors[i*2][j] - parameters[1]*eigenVectors[(i*2)+1][j] + eigenVectors[i*2][j];
            j1 = parameters[0]*eigenVectors[(i*2)+1][j] + parameters[1]*eigenVectors[i*2][j] + eigenVectors[(i*2)+1][j];
            jacobian[i][j+4] = j0;
            jacobian[i+numPatches][j+4] = j1;
    return jacobian

# faster version of createJacobian
def createJacobian2(parameters, meanXShape, meanYShape, xEigenVectors, yEigenVectors):
    numParameters = len(parameters)-4
    numPatches = len(meanXShape)
    jacobian = np.zeros((2*numPatches, numParameters+4))
    j0 = meanXShape + np.dot(xEigenVectors, parameters[4:]).reshape(numPatches, 1)
    j1 = meanYShape + np.dot(yEigenVectors, parameters[4:]).reshape(numPatches, 1)
    jacobian[0:numPatches, 0:1] = j0
    jacobian[numPatches:numPatches*2, 0:1] = j1
    jacobian[0:numPatches, 1:2] = -j1
    jacobian[numPatches:numPatches*2, 1:2] = j0
    jacobian[0:numPatches, 2:3] = np.ones((numPatches, 1))
    jacobian[numPatches:numPatches*2, 3:4] = np.ones((numPatches, 1))
    jacobian[0:numPatches, 4:] = xEigenVectors * (parameters[0]+1) - yEigenVectors * parameters[1]
    jacobian[numPatches:numPatches*2, 4:] = xEigenVectors * parameters[1] + yEigenVectors * (parameters[0]+1)
    return jacobian

def gpopt(responseWidth, currentPositionsj, vecProbs,
          responses, opj0, opj1, j, variance, scaling):
    pos_idx = 0;
    vpsum = 0;
    for k in range(responseWidth):
        updatePositionY = opj1+(k*scaling);
        for l in range(responseWidth):
            updatePositionX = opj0+(l*scaling);
            dx = currentPositionsj[0] - updatePositionX
            dy = currentPositionsj[1] - updatePositionY;
            vecProbs[pos_idx] = responses[j][pos_idx] * math.exp(-0.5*((dx*dx)+(dy*dy))/(variance*scaling));
            vpsum += vecProbs[pos_idx];
            pos_idx+=1;
    return vpsum;

def gpopt2(responseWidth, vecpos, vecProbs, vpsum, opj0, opj1, scaling):
    pos_idx = 0;
    vecsum = 0;
    vecpos[0] = 0;
    vecpos[1] = 0;
    for k in range(responseWidth):
        updatePositionY = opj1+(k*scaling);
        for l in range(responseWidth):
            updatePositionX = opj0+(l*scaling);
            vecsum = vecProbs[pos_idx]/vpsum;
            vecpos[0] += vecsum*updatePositionX;
            vecpos[1] += vecsum*updatePositionY;
            pos_idx+=1;

# fast way of calculating meanshift in original position domain
def getMeanShift2(responses, searchWindow, weightArray, indXYArray, scaling, variance):
    numPatches = len(responses)
    vecProbs = np.multiply(responses, weightArray)
    vecAvg = (1.0/np.sum(vecProbs, axis=1)).reshape(numPatches, 1)
    vecProbs = np.multiply(vecProbs, vecAvg)
    xyShift = np.dot(vecProbs, indXYArray*scaling)
    return xyShift

# fast way of calculating meanshift in patch domain
def getMeanShift(responses, parameters, searchWindow, indX2Y2Array, indXYArray, scaling, variance):
    numPatches = len(responses)
    applyWeight = lambda x: math.exp(-0.5*x*scaling/variance)
    applyWeightFunc = np.vectorize(applyWeight)
    weightArray = applyWeightFunc(indX2Y2Array)
    vecProbs = np.multiply(responses, weightArray)
    vecAvg = (1.0/np.sum(vecProbs, axis=1)).reshape(numPatches, 1)
    vecProbs = np.multiply(vecProbs, vecAvg)
    xyShift = np.dot(vecProbs, indXYArray)
    M = np.array([[parameters[0]+1, parameters[1]],
                  [-parameters[1] , parameters[0]+1]])
    xyShift = np.dot(xyShift, M)
    return xyShift

def getConvergence(previousPositions):
    if len(previousPositions) < 10:
        return 999999
    prevX = 0.0
    prevY = 0.0
    currX = 0.0
    currY = 0.0
    numPatches = len(previousPositions[0])
    for i in range(5):
        for j in range(numPatches):
            prevX += previousPositions[i][j][0];
            prevY += previousPositions[i][j][1];
    prevX /= 5.0
    prevY /= 5.0
    for i in range(5, 10):
        for j in range(numPatches):
            currX += previousPositions[i][j][0];
            currY += previousPositions[i][j][1];
    currX /= 5.0
    currY /= 5.0
    diffX = currX-prevX;
    diffY = currY-prevY;
    msavg = ((diffX*diffX) + (diffY*diffY));
    msavg /= (1.0*len(previousPositions));
    return msavg;
