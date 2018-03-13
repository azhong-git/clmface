import os
import sys
import logging
import json
import math
from collections import deque
from copy import deepcopy
import time

from dotmap import DotMap
import matplotlib.pyplot as plt
import cv2
import numpy as np

from lib.math_lib import procrustes, logistic, normalize
from lib.tracking import getInitialPosition, calculatePositions, calculatePositions2, createJacobian, createJacobian2, gpopt, gpopt2, gpopt_new, getConvergence, convertPatchVectorToPositionVector, getMeanShift, getMeanShift2
from lib.image import getImageData

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../models/')
logging.basicConfig(level=logging.DEBUG)

class clmFaceTracker:
    def __init__(self, model_path, debug=False):
        if not os.path.isfile(model_path):
            logging.error('model {} not found'.format(model_path))
            sys.exit(1)
        else:
            with open(model_path) as fi:
                self.model = DotMap(json.load(fi))
        self._config()
        self.debug = debug

        self._initial_state()

    # configure all the non-change variables
    def _config(self):
        config = {
            'constantVelocity': True,
            'searchWindow': 11,
            'scoreThreshold': 0.5,
            'stopOnConvergence': False,
            'weightPoints': None,
            'sharpenResponse': False,
            'convergenceLimit': 0.01,
            'convergenceThreshold': 0.5,
            'velocityInterpolation': 0.9
        }
        self.config = DotMap(config)

        # model based configurations
        self.config.patchType = self.model.patchModel.patchType
        self.config.numPatches = self.model.patchModel.numPatches
        self.config.patchSize = self.model.patchModel.patchSize[0]
        self.config.modelWidth = self.model.patchModel.canvasSize[0]
        self.config.modelHeight = self.model.patchModel.canvasSize[1]
        self.config.weights = self.model.patchModel.weights
        self.config.biases = self.model.patchModel.bias

        if self.config.patchType == 'MOSSE':
            self.config.searchWindow = self.model.patchModel.patchSize[0]
        else:
            assert self.config.patchType == 'SVM'
        self.config.numParameters = self.model.shapeModel.numEvalues
        self.config.sketchW = self.config.modelWidth + self.config.searchWindow-1 + self.config.patchSize-1
        self.config.sketchH = self.config.modelHeight + self.config.searchWindow-1 + self.config.patchSize-1

        # eigenVectors of shape (numPatches*2, numParameters) (71*2, 20)
        self.config.eigenVectors = self.model.shapeModel.eigenVectors
        self.config.xEigenVectors = []
        self.config.yEigenVectors = []
        # compute x and y eigenVectors
        for i in range(self.config.numPatches):
            # x/yEigenvectors of shape (numPatches, numParameters) (71, 20)
            self.config.xEigenVectors.append(self.config.eigenVectors[i*2])
            self.config.yEigenVectors.append(self.config.eigenVectors[i*2+1])
        self.config.xEigenVectors = np.array(self.config.xEigenVectors)
        self.config.yEigenVectors = np.array(self.config.yEigenVectors)
        self.config.eigenValues = self.model.shapeModel.eigenValues
        self.config.meanShape = np.array(self.model.shapeModel.meanShape)
        self.config.meanXShape = []
        self.config.meanYShape = []
        for i in range(self.config.numPatches):
            self.config.meanXShape.append([self.config.meanShape[i][0]])
            self.config.meanYShape.append([self.config.meanShape[i][1]])
        self.config.meanXShape = np.array(self.config.meanXShape)
        self.config.meanYShape = np.array(self.config.meanYShape)
        self.config.msmodelwidth = max(self.config.meanShape.T[0]) - min(self.config.meanShape.T[0])
        self.config.msmodelheight = max(self.config.meanShape.T[1]) - min(self.config.meanShape.T[1])

        # runtime configurations
        self.config.varianceSeq = [10, 5, 1]
        logistic_func = lambda x: 1.0/(1.0+math.exp(-(x-1)))
        self.config.logistic_func = np.vectorize(logistic_func)
        # faster meanshift
        self.config.indXArray = np.zeros(self.config.searchWindow*self.config.searchWindow)
        self.config.indYArray = np.zeros(self.config.searchWindow*self.config.searchWindow)
        self.config.indXYArray = np.zeros(self.config.searchWindow*self.config.searchWindow)
        for i in range(self.config.searchWindow):
            for j in range(self.config.searchWindow):
                self.config.indXArray[i*self.config.searchWindow+j] = j-self.config.searchWindow/2
                self.config.indYArray[i*self.config.searchWindow+j] = i-self.config.searchWindow/2
                self.config.indXYArray[i*self.config.searchWindow+j] = (i-self.config.searchWindow/2)**2+(j-self.config.searchWindow/2)**2

    def _initial_state(self):
        self.vecpos = [0.0, 0.0]
        self.vecProbs = np.zeros(self.config.searchWindow * self.config.searchWindow)
        self.gaussianPD = np.zeros((self.config.numParameters+4, self.config.numParameters+4))
        for i in range(self.config.numParameters):
            if i in self.model.shapeModel.nonRegularizedVectors:
                self.gaussianPD[i+4][i+4] = 1.0/1e7
            else:
                self.gaussianPD[i+4][i+4] = 1.0/self.config.eigenValues[i]
        self.currentParameters = np.zeros(self.config.numParameters+4)
        # in full-size coordinates
        self.currentPositions = None
        # in scaled-down and rotated coordiantes
        self.patchPositions = None

        # history of states
        self.previousPositions = deque(maxlen=10)
        self.scoringHistory = deque(maxlen=5)
        self.previousParameters = deque(maxlen=2)

        # check if currently a face is in track
        self.detected_face = False
        return

    def track(self, gray):
        shape = gray.shape
        assert len(shape) == 2, 'only track grayscale image'
        if not self.detected_face:
            start = time.time()
            # useful if the grayscale image contains floating point numbers
            gray_truncated = np.asarray(gray, dtype=np.uint8)
            self.detected_face, (translateX, translateY, scaling, rotation) = getInitialPosition(
                gray_truncated, self.model,
                modelDir, debug=self.debug)
            if not self.detected_face:
                return False
            # AZ: tuning why -1?
            self.currentParameters[0] = scaling*math.cos(rotation)-1
            self.currentParameters[1] = scaling*math.sin(rotation)
            self.currentParameters[2] = translateX
            self.currentParameters[3] = translateY

            self.patchPositions, self.currentPositions = calculatePositions2(self.currentParameters,
                                                                             self.config.meanXShape, self.config.meanYShape,
                                                                             self.config.xEigenVectors, self.config.yEigenVectors, True)
            logging.debug('initial detection takes {} ms'.format((time.time() - start)*1e3))
            if self.debug:
                logging.debug( 'currentParameters init: {}'.format(self.currentParameters))
                dst = gray.copy()
                for x, y in self.currentPositions:
                    cv2.circle(dst, (int(x), int(y)), 2, (255))
                plt.imshow(dst, cmap='gray')
                plt.title('fine-grained face landmarks based on initial guess')
                plt.show()
            return self.refine_tracking(gray, tracking=False)
        else:
            return self.refine_tracking(gray, tracking=True)

    def refine_tracking(self, gray, tracking=False):
        iteration = 0
        if True:
            iteration += 1
            # AZ: what about rotation == 0
            start = time.time()
            if self.config.constantVelocity and len(self.previousParameters) >= 2:
                self.currentParameters = -self.config.velocityInterpolation*self.previousParameters[0]+(1+self.config.velocityInterpolation)*self.previousParameters[1]
            assert self.currentParameters[1] != 0
            rotation = math.pi/2 - math.atan((self.currentParameters[0]+1.0)/self.currentParameters[1])
            if rotation > math.pi/2:
                rotation -= math.pi
            scaling = self.currentParameters[1]/math.sin(rotation)
            translateX = self.currentParameters[2]
            translateY = self.currentParameters[3]
            rows, cols = gray.shape
            M = cv2.getRotationMatrix2D((translateX, translateY), rotation/math.pi*180, 1)
            M[0, 2] -= translateX
            M[1, 2] -= translateY
            current_gray = cv2.warpAffine(gray, M, (cols, rows))
            # current_gray = cv2.warpAffine(gray, M, (2*cols, 2*rows))
            logging.debug('rotation/translation takes {} ms'.format((time.time() - start)*1e3))
            start = time.time()
            # M = np.float32([[1.0, 0, -translateX],
            #                 [0, 1.0, -translateY]])
            # current_gray = cv2.warpAffine(current_gray, M, (cols, rows))
            current_gray = cv2.resize(current_gray, None, 0, 1/scaling, 1/scaling, interpolation=cv2.INTER_NEAREST)
            current_gray = current_gray[0:self.config.sketchH, 0:self.config.sketchW]
            logging.debug('scale takes {} ms'.format((time.time() - start)*1e3))

            if self.debug >= 4 :
                logging.debug('currentParameters are {}'.format(self.currentParameters))
                plt.imshow(current_gray, cmap='gray')
                plt.title('scaled and transformed to intial model size')
                plt.show()
                current_gray_dst = current_gray.copy()
                for x, y in self.patchPositions:
                    cv2.circle(current_gray_dst, (x, y), 2, (255))
                plt.imshow(current_gray_dst, cmap='gray')
                plt.title('fine-grained face landmarks on the small image')
                plt.show()


            # if update currentParameters with constantVelocity assumption, need to recompute patchPositions
            if self.config.constantVelocity:
                self.patchPositions, _ =  calculatePositions2(
                    self.currentParameters, self.config.meanXShape, self.config.meanYShape,
                    self.config.xEigenVectors, self.config.yEigenVectors, False)

            if self.config.patchType == 'SVM':
                start = time.time()
                pw = self.config.patchSize+self.config.searchWindow-1
                pl = self.config.patchSize+self.config.searchWindow-1
                debug_patches = []
                patches = []
                responses = []
                for i in range(self.config.numPatches):
                    patch = getImageData(current_gray, self.patchPositions[i][0], self.patchPositions[i][1],
                                         pw, pl)
                    # normalize (alternative way, way faster than math_lib.normalize)
                    maxv, minv = np.max(patch), np.min(patch)
                    patch = (patch - minv) / 1.0 / (maxv - minv)
                    patches.append(patch)
                logging.debug('preparing patches takes {} ms'.format((time.time() - start)*1e3))

                # raw filter responses
                start = time.time()
                for i in range(self.config.numPatches):
                    kernel = np.asarray(self.config.weights['raw'][i]).reshape(self.config.patchSize,
                                                                               self.config.patchSize)
                    dst_d = cv2.filter2D(patches[i], -1, kernel, delta=self.config.biases['raw'][i])
                    response = dst_d[self.config.patchSize/2:-(self.config.patchSize/2),
                                     self.config.patchSize/2:-(self.config.patchSize/2)]
                    # apply logistic function (alternative way, faster than math_lib.logistic)
                    response = self.config.logistic_func(response)

                    # normalize (alternative way, way faster than math_lib.normalize)
                    maxv, minv = np.max(response), np.min(response)
                    response = (response - minv) / 1.0 / (maxv - minv)

                    response = response.reshape(self.config.searchWindow*self.config.searchWindow)

                    responses.append(response.copy())
                if self.config.sharpenResponse:
                    responses = np.power(responses, self.config.sharpenResponse)
                logging.debug('applying filters takes {} ms'.format((time.time() - start)*1e3))

                originalPositions = deepcopy(self.currentPositions)
                iteration_minor = 0
                responses = np.array(responses)

                for i in range(len(self.config.varianceSeq)):
                    start = time.time()
                    iteration_minor += 1
                    jac = createJacobian2(self.currentParameters, self.config.meanXShape, self.config.meanYShape,
                                          self.config.xEigenVectors, self.config.yEigenVectors)
                    logging.debug('jacobian creation takes {} ms'.format((time.time()-start)*1e3))
                    start = time.time()

                    meanShiftVector = np.zeros((self.config.numPatches*2, 1))
                    ## AZ: old way of doing meanshift, I consider it wrong (no rotation applied and slow)
                    ## faster version of this is implemented in getMeanShift2
                    # for j in range(self.config.numPatches):
                    #     opj0 = originalPositions[j][0] - ((self.config.searchWindow-1)*scaling/2)
                    #     opj1 = originalPositions[j][1] - ((self.config.searchWindow-1)*scaling/2)
                    #     vpsum = gpopt(self.config.searchWindow, self.currentPositions[j], self.vecProbs,
                    #                   responses, opj0, opj1, j, self.config.varianceSeq[i], scaling)
                    #     gpopt2(self.config.searchWindow, self.vecpos, self.vecProbs, vpsum, opj0, opj1, scaling)
                    #     meanShiftVector[j] = self.vecpos[0] - self.currentPositions[j][0]
                    #     meanShiftVector[j+self.config.numPatches] = self.vecpos[1] - self.currentPositions[j][1]
                    xyShift = getMeanShift2(responses, self.currentParameters,
                                           self.config.searchWindow, self.config.indXYArray,
                                           self.config.indXArray, self.config.indYArray,
                                           scaling, self.config.varianceSeq[i])
                    meanShiftVector[0:self.config.numPatches]=xyShift[:, [0]]
                    meanShiftVector[self.config.numPatches:2*self.config.numPatches]=xyShift[:, [1]]
                    logging.debug('calculating meanshift takes {} ms'.format((time.time() - start)*1e3))
                    start = time.time()

                    if self.debug:
                        dst = gray.copy()
                        for j in range(self.config.numPatches):
                            x, y = self.currentPositions[j]
                            x_new = x + meanShiftVector[j][0]
                            y_new = y + meanShiftVector[j+self.config.numPatches][0]
                            cv2.circle(dst, (int(x_new), int(y_new)), 2, (0, 255, 0))
                            cv2.circle(dst, (int(x), int(y)), 2, (255, 0, 0))
                            cv2.line(dst, (int(x),int(y)), (int(x_new), int(y_new)), (255, 255, 255), 2)
                        plt.imshow(dst, cmap='gray')
                        plt.title('meanshift iteration {}.{}'.format(iteration, iteration_minor))
                        plt.show()

                    # compute pdm paramter update
                    prior = np.dot(self.gaussianPD, self.config.varianceSeq[i])
                    jtj = np.dot(np.transpose(jac), jac)
                    cpMatrix = np.zeros((self.config.numParameters+4, 1))
                    for l in range(self.config.numParameters+4):
                        cpMatrix[l][0] = self.currentParameters[l]
                    priorP = np.dot(prior, cpMatrix)
                    jtv = np.dot(np.transpose(jac), meanShiftVector)
                    paramUpdateLeft = np.add(prior, jtj)
                    paramUpdateRight = np.subtract(priorP, jtv)
                    paramUpdate = np.dot(np.linalg.inv(paramUpdateLeft), paramUpdateRight)
                    oldPositions = deepcopy(self.currentPositions)
                    logging.debug('pdm parameter update takes {} ms'.format((time.time() - start)*1e3))
                    start = time.time()

                    # update estimated parameters
                    self.currentParameters -= paramUpdate.reshape(self.config.numParameters+4)
                    # clipping of parameters if they are too high
                    for k in range(self.config.numParameters):
                        clip = math.fabs(3*math.sqrt(self.config.eigenValues[k]))
                        if math.fabs(self.currentParameters[k+4])>clip:
                            if self.currentParameters[k+4]>0:
                                self.currentParameters[k+4]=clip
                            else:
                                self.currentParameters[k+4]=-clip
                    logging.debug('updating currentParameters takes {} ms'.format((time.time()-start)*1e3))
                    start = time.time()
                    self.patchPositions, self.currentPositions = calculatePositions2(self.currentParameters, self.config.meanXShape, self.config.meanYShape,
                                                                                     self.config.xEigenVectors, self.config.yEigenVectors, True)
                    logging.debug('updating positions for iteration {}.{} takes {} ms'.format(iteration, iteration_minor,
                                                                                      (time.time() - start)*1e3))

                    if self.debug >= 2:
                        dst = gray.copy()
                        for x, y in self.currentPositions:
                            cv2.circle(dst, (int(x), int(y)), 2, (255))
                        plt.imshow(dst, cmap='gray')
                        plt.title('fine-grained face landmarks first iteration')
                        plt.show()
                    positionNorm = 0
                    for k in range(len(self.currentPositions)):
                        pnsq_x = self.currentPositions[k][0] - oldPositions[k][0]
                        pnsq_y = self.currentPositions[k][1] - oldPositions[k][1]
                        positionNorm += pnsq_x * pnsq_x + pnsq_y * pnsq_y
                    logging.info('iteration {}.{}: position norm is {}, limit is {}'.format(iteration, iteration_minor, positionNorm, self.config.convergenceLimit))

                    if positionNorm < self.config.convergenceLimit:
                        break

                self.previousPositions.append(self.currentPositions)
                if self.config.constantVelocity:
                    self.previousParameters.append(self.currentParameters)
                convergenceScore = getConvergence(self.previousPositions)
                logging.info('iteration {}.{}: convergence score is {}'.format(iteration, iteration_minor, convergenceScore))
                if convergenceScore < self.config.convergenceThreshold:
                    logging.info('CLM tracker converged: score is {} < {}'.format(convergenceScore, self.config.convergenceLimit))
                    return True
                return False

def main():
    if len(sys.argv) < 2:
        logging.error('no model specified')
        sys.exit(1)
    tracker = clmFaceTracker(sys.argv[1], debug=False)

    imagePath = os.path.join(fileDir, '../examples/images/franck_02159.jpg')
    img = cv2.imread(imagePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (625, 500))
    # gray=img[:,:,0]*0.3+img[:,:,1]*0.59+img[:,:,2]*0.11
    # the following conversion to integer pixel val makes a lot of arithmetic faster
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # plt.imshow(gray, cmap='gray')
    # plt.show()
    iteration = 0
    while True:
        converged = tracker.track(gray)
        iteration += 1
        if converged:
            print 'CLMTracker converged after {} iterations'.format(iteration)
            break

    #    tracker.track

if __name__ == "__main__":
    main()
