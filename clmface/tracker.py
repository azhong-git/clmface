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
from lib.tracking import getInitialPosition, calculatePositions, calculatePositions2, createJacobian, createJacobian2, gpopt, gpopt2, getConvergence, convertPatchVectorToPositionVector, getMeanShift, getMeanShift2, check_face_score
from lib.image import getImageData

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../models/')
logging.basicConfig(level=logging.INFO)

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
        weightPoints = [1.0 for i in range(self.model.patchModel.numPatches)]
        eyeIndices = [23,24,25,26,27,28,29,30,31,32,63,64,65,66,67,68,69,70]
        mouthIndices = [44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61]
        mouthOpenIndices = [59, 60, 61, 56, 57, 58, 52, 53, 54]
        mouthOpenIndices2 = [59, 60, 61, 56, 57, 58]
        jawIndices = [4,5,6,7,8,9,10]
        for i in eyeIndices:
            weightPoints[i] = 1.0
        for i in mouthIndices:
            weightPoints[i] = 1.0
        for i in mouthOpenIndices2:
            weightPoints[i] = 0.2
        for i in jawIndices:
            weightPoints[i] = 1.0
        config = {
            'constantVelocity': True,
            'searchWindow': 11,
            'scoreThreshold': 0.5,
            'stopOnConvergence': False,
            'weightPoints': weightPoints,
            'sharpenResponse': False,
            'convergenceLimit': 0.01,
            'convergenceThreshold': 0.5,
            'velocityInterpolation': 0.9,
            'localTracking': True
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
        self.config.scoringWeights = self.model.scoring.coef
        self.config.scoringBias = self.model.scoring.bias

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
        self.config.msxmin = min(self.config.meanShape.T[0])
        self.config.msxmax = max(self.config.meanShape.T[0])
        self.config.msymin = min(self.config.meanShape.T[1])
        self.config.msymax = max(self.config.meanShape.T[1])
        self.config.msmodelwidth = self.config.msxmax - self.config.msxmin
        self.config.msmodelheight = self.config.msymax - self.config.msymin

        # check face fit score with a 20x22 window filter
        self.config.score_start_x = int(round(self.config.msxmin+(self.config.msmodelwidth/4.5)))
        self.config.score_start_y = int(round(self.config.msymin-(self.config.msmodelheight/12.0)))
        self.config.score_end_x = int(round(self.config.msmodelwidth - (self.config.msmodelwidth*2/4.5))) + self.config.score_start_x
        self.config.score_end_y = int(round(self.config.msmodelheight - (self.config.msmodelheight/12.0))) + self.config.score_start_y
        self.config.score_patch_width = 20
        self.config.score_patch_height = 22

        # runtime configurations
        self.config.varianceSeq = [10, 5, 1]
        logistic_func = lambda x: 1.0/(1.0+math.exp(-(x-1)))
        self.config.logistic_func = np.vectorize(logistic_func)
        # faster meanshift
        self.config.indXYArray = np.zeros((self.config.searchWindow*self.config.searchWindow, 2))
        self.config.indNXArray = np.zeros((self.config.numPatches, self.config.searchWindow*self.config.searchWindow))
        self.config.indNYArray = np.zeros((self.config.numPatches, self.config.searchWindow*self.config.searchWindow))
        self.config.indX2Y2Array = np.zeros(self.config.searchWindow*self.config.searchWindow)
        for i in range(self.config.searchWindow):
            for j in range(self.config.searchWindow):
                self.config.indXYArray[i*self.config.searchWindow+j, 0] = j-self.config.searchWindow/2
                self.config.indXYArray[i*self.config.searchWindow+j, 1] = i-self.config.searchWindow/2
                self.config.indNXArray[:, i*self.config.searchWindow+j] = (j-self.config.searchWindow/2)
                self.config.indNYArray[:, i*self.config.searchWindow+j] = (i-self.config.searchWindow/2)
                self.config.indX2Y2Array[i*self.config.searchWindow+j] = (j-self.config.searchWindow/2)**2+(i-self.config.searchWindow/2)**2
        if self.config.weightPoints:
            pointWeight = deepcopy(self.config.weightPoints)
            pointWeight.extend(pointWeight)
            self.config.pointWeight = np.diag(pointWeight)

        self.config.trackWindow = 20 # track a 20 pixel window for local tracker

        # for debug only
        self.config.debug = False
        self.config.debugPatches = None
        self.config.debugCurrentParameters = None
        self.config.debugCurrentPositions = None

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

        # tracking
        if self.config.localTracking:
            self.localTracker = cv2.MultiTracker_create()
            self.lastFrameLocallyTracked = False
            self.localTrackerUpdate = None

        # check if currently a face is in track
        self.detected_face = False
        self.max_face_score = 0
        self.iteration = 0
        return

    def resetParameters(self):
        self.detected_face = False
        self.max_face_score = 0
        self.currentParameters = np.zeros(self.config.numParameters+4)
        # in full-size coordinates
        self.currentPositions = None
        self.previousPositions = deque(maxlen=10)
        self.scoringHistory = deque(maxlen=5)
        self.previousParameters = deque(maxlen=2)
        self.iteration = 0

    def track(self, gray):
        shape = gray.shape
        assert len(shape) == 2, 'only track grayscale image'
        if not self.detected_face:
            start = time.time()
            # useful if the grayscale image contains floating point numbers
            gray_truncated = np.asarray(gray, dtype=np.uint8)
            self.detected_face, (translateX, translateY, scaling, rotation), trackPoints = getInitialPosition(
                gray_truncated, self.model,
                modelDir, debug=self.debug)
            if not self.detected_face:
                return False, 0
            # AZ: tuning why -1?
            self.currentParameters[0] = scaling*math.cos(rotation)-1
            self.currentParameters[1] = scaling*math.sin(rotation)
            self.currentParameters[2] = translateX
            self.currentParameters[3] = translateY

            self.patchPositions, self.currentPositions = calculatePositions2(self.currentParameters,
                                                                             self.config.meanXShape, self.config.meanYShape,
                                                                             self.config.xEigenVectors, self.config.yEigenVectors, True)
            logging.debug('initial detection takes {:.2f} ms'.format((time.time() - start)*1e3))

            if self.config.localTracking:
                self.localTracker = cv2.MultiTracker_create()
                self.localTrackerRects = []
                for [x, y] in trackPoints:
                    rect = (int(x)-self.config.trackWindow/2, int(y)-self.config.trackWindow/2,
                            self.config.trackWindow, self.config.trackWindow)
                    ok = self.localTracker.add(cv2.TrackerMedianFlow_create(), gray, rect)

            if self.debug:
                logging.debug( 'currentParameters init: {}'.format(self.currentParameters))
                dst = gray.copy()
                for x, y in self.currentPositions:
                    cv2.circle(dst, (int(x), int(y)), 2, (255))
                plt.imshow(dst, cmap='gray')
                plt.title('fine-grained face landmarks based on initial guess')
                plt.show()
            return self.refine_tracking(gray, initTracking=True)
        else:
            return self.refine_tracking(gray, initTracking=False)

    def start_timer(self):
        self.time_last = time.time()
        self.time_total = 0

    def update_timer(self, event_name):
        time_now = time.time()
        self.time_total += (time_now - self.time_last)*1e3
        logging.debug('{} takes {:.2f}ms, {:.2f}ms'.format(event_name,
                                                           (time_now-self.time_last)*1e3,
                                                           self.time_total))
        self.time_last = time.time()


    def refine_tracking(self, gray, initTracking=False):
        if self.config.debug:
            self.currentParameters = self.config.debugCurrentParameters
            self.currentPositions = self.config.debugCurrentPositions

        self.start_timer()
        self.iteration += 1
        if self.config.localTracking:
            lastFrameLocallyTracked = self.lastFrameLocallyTracked
            ok, rects = self.localTracker.update(gray)
            if ok:
                self.localTrackerRects = rects
                assert len(self.localTrackerRects) == 3, 'all features need to be tracked'
                if lastFrameLocallyTracked:
                    lastTrackerRectCenters = self.localTrackerRectCenters
                    self.localTrackerRectCenters = ([[x+w/2, y+h/2] for x,y,w,h in rects])
                    translateX, translateY, scaling, rotation = procrustes(self.localTrackerRectCenters,
                                                                           lastTrackerRectCenters)
                    self.localTrackerUpdate = [translateX, translateY, scaling, rotation]
                    M = np.array([[self.currentParameters[0]+1, -self.currentParameters[1], self.currentParameters[2]],
                                  [self.currentParameters[1] , self.currentParameters[0]+1, self.currentParameters[3]],
                                  [0.0, 0.0, 1]])
                    Mupdate = np.array([[scaling*math.cos(rotation), -scaling*math.sin(rotation), translateX],
                                        [scaling*math.sin(rotation), scaling*math.cos(rotation), translateY],
                                        [0.0, 0.0, 1.0]])
                    M = np.dot(Mupdate, M)
                    self.currentParameters[0] = M[0][0] - 1
                    self.currentParameters[1] = M[1][0]
                    self.currentParameters[2] = M[0][2]
                    self.currentParameters[3] = M[1][2]
                else:
                    self.localTrackerRectCenters = ([[x+w/2, y+h/2] for x,y,w,h in rects])
                    self.localTrackerUpdate = None

                self.lastFrameLocallyTracked = True
            else:
                self.lastFrameLocallyTracked = False
                self.localTrackerUpdate = None
            self.update_timer('localTracking')

        if self.config.constantVelocity and len(self.previousParameters) >= 2:
            if self.config.localTracking and lastFrameLocallyTracked:
                self.currentParameters[4:] = -self.config.velocityInterpolation*self.previousParameters[0][4:]+(1+self.config.velocityInterpolation)*self.previousParameters[1][4:]
            else:
                self.currentParameters = -self.config.velocityInterpolation*self.previousParameters[0]+(1+self.config.velocityInterpolation)*self.previousParameters[1]

        # AZ: what about rotation == 0
        assert self.currentParameters[1] != 0
        rotation = math.pi/2 - math.atan((self.currentParameters[0]+1.0)/self.currentParameters[1])
        if rotation > math.pi/2:
            rotation -= math.pi
        scaling = self.currentParameters[1]/math.sin(rotation)
        translateX = self.currentParameters[2]
        translateY = self.currentParameters[3]
        if scaling < 0.5:
            self.resetParameters()
            logging.info('face is too small, reinitialize')
            return False, 0
        rows, cols = gray.shape
        M = cv2.getRotationMatrix2D((translateX, translateY), rotation/math.pi*180, 1)
        M[0, 2] -= translateX
        M[1, 2] -= translateY
        current_gray = cv2.warpAffine(gray, M, (cols, rows))
        # current_gray = cv2.warpAffine(gray, M, (2*cols, 2*rows))
        self.update_timer('rotation/translation')

        # M = np.float32([[1.0, 0, -translateX],
        #                 [0, 1.0, -translateY]])
        # current_gray = cv2.warpAffine(current_gray, M, (cols, rows))
        current_gray = cv2.resize(current_gray, None, 0, 1.0/scaling, 1.0/scaling)
        current_gray = current_gray[0:self.config.sketchH, 0:self.config.sketchW]
        self.update_timer('scaling')

        # check face score
        current_gray_scoring = current_gray[self.config.score_start_y: self.config.score_end_y,
                                            self.config.score_start_x: self.config.score_end_x]
        current_gray_scoring = cv2.resize(current_gray_scoring, (self.config.score_patch_width, self.config.score_patch_height)).reshape(self.config.score_patch_width*self.config.score_patch_height)
        face_score = check_face_score(current_gray_scoring, self.config.scoringWeights, self.config.scoringBias)
        self.scoringHistory.append(face_score)
        if face_score > self.max_face_score:
            self.max_face_score = face_score
        self.update_timer('face scoring')
        mean_score = np.mean(self.scoringHistory)
        if mean_score < self.max_face_score*0.4 or mean_score < 0.2 or (self.iteration > 5 and self.max_face_score < 0.5):
            self.resetParameters()
            return False, 0

        # if update currentParameters with constantVelocity assumption, need to recompute patchPositions
        if self.config.constantVelocity:
            self.patchPositions, _ =  calculatePositions2(
                self.currentParameters, self.config.meanXShape, self.config.meanYShape,
                self.config.xEigenVectors, self.config.yEigenVectors, False)
            self.update_timer('updating patchPositions because of constVelocity')

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
            plt.imshow(current_gray_scoring.reshape((self.config.score_patch_height, self.config.score_patch_width)), cmap='gray')
            plt.title('scoring context')
            plt.show()


        if self.config.patchType == 'SVM':
            responses = []
            pw = self.config.patchSize+self.config.searchWindow-1
            pl = self.config.patchSize+self.config.searchWindow-1
            if self.config.debug:
                patches = self.config.debugPatches
                for i in range(self.config.numPatches):
                    maxv, minv = np.max(patches[i]), np.min(patches[i])
                    patches[i] = (patches[i] - minv) / 1.0 / (maxv - minv)
            else:
                patches = []
                for i in range(self.config.numPatches):
                    patch = getImageData(current_gray, self.patchPositions[i][0], self.patchPositions[i][1],
                                         pw, pl)
                    # normalize (alternative way, way faster than math_lib.normalize)
                    maxv, minv = np.max(patch), np.min(patch)
                    if maxv - minv < 1e-10:
                        patch = patch - minv
                    else:
                        patch = (patch - minv) / 1.0 / (maxv - minv)
                    patches.append(patch)
                self.update_timer('preparing patches')
            # raw filter responses
            for i in range(self.config.numPatches):
                kernel = np.asarray(self.config.weights['raw'][i]).reshape(self.config.patchSize,
                                                                           self.config.patchSize)
                dst_d = cv2.filter2D(patches[i], -1, kernel, delta=self.config.biases['raw'][i])
                response = dst_d[self.config.patchSize/2:-(self.config.patchSize/2),
                                 self.config.patchSize/2:-(self.config.patchSize/2)]

                # apply logistic function (alternative way, faster than math_lib.logistic)
                # response = self.config.logistic_func(response)
                response = 1.0/(1.0+np.exp(1-response))

                # normalize (alternative way, way faster than math_lib.normalize)
                maxv, minv = np.max(response), np.min(response)
                response = (response - minv) / 1.0 / (maxv - minv)
                response = response.reshape(self.config.searchWindow*self.config.searchWindow)

                responses.append(response.copy())
            if self.config.sharpenResponse:
                responses = np.power(responses, self.config.sharpenResponse)
            self.update_timer('applying filters')

            originalPositions = deepcopy(self.currentPositions)
            iteration_minor = 0
            responses = np.array(responses)

            for i in range(len(self.config.varianceSeq)):
                iteration_minor += 1
                jac = createJacobian2(self.currentParameters, self.config.meanXShape, self.config.meanYShape,
                                      self.config.xEigenVectors, self.config.yEigenVectors)
                self.update_timer('creating jacobian')

                meanShiftVector = np.zeros((self.config.numPatches*2, 1))
                #AZ: old way of doing meanshift, I consider it wrong (no rotation applied and slow)
                #faster version of this is implemented in getMeanShift2
                # for j in range(self.config.numPatches):
                #     opj0 = originalPositions[j][0] - ((self.config.searchWindow-1)*scaling/2)
                #     opj1 = originalPositions[j][1] - ((self.config.searchWindow-1)*scaling/2)
                #     vpsum = gpopt(self.config.searchWindow, self.currentPositions[j], self.vecProbs,
                #                   responses, opj0, opj1, j, self.config.varianceSeq[i], scaling)
                #     gpopt2(self.config.searchWindow, self.vecpos, self.vecProbs, vpsum, opj0, opj1, scaling)
                #     meanShiftVectorOriginal[j] = self.vecpos[0] - self.currentPositions[j][0]
                #     meanShiftVectorOriginal[j+self.config.numPatches] = self.vecpos[1] - self.currentPositions[j][1]

                dxy = originalPositions - self.currentPositions
                dx2 = np.power(np.add(self.config.indNXArray*scaling, dxy[:, [0]]), 2)
                dy2 = np.power(np.add(self.config.indNYArray*scaling, dxy[:, [1]]), 2)
                weightArray = np.exp((dx2+dy2)*(-0.5/self.config.varianceSeq[i]/scaling))
                xyShift = getMeanShift2(responses,
                                       self.config.searchWindow,
                                       weightArray,
                                       self.config.indXYArray,
                                       scaling, self.config.varianceSeq[i])
                xyShift += dxy
                meanShiftVector[0:self.config.numPatches]=xyShift[:, [0]]
                meanShiftVector[self.config.numPatches:2*self.config.numPatches]=xyShift[:, [1]]
                self.update_timer('calculating meanshift')

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
                    plt.title('meanshift iteration {}'.format(iteration_minor))
                    plt.show()

                # compute pdm paramter update
                prior = np.dot(self.gaussianPD, self.config.varianceSeq[i])
                if (self.config.weightPoints):
                    jtj = np.dot(np.transpose(jac), np.dot(self.config.pointWeight, jac))
                else:
                    jtj = np.dot(np.transpose(jac), jac)
                cpMatrix = np.zeros((self.config.numParameters+4, 1))
                for l in range(self.config.numParameters+4):
                    cpMatrix[l][0] = self.currentParameters[l]
                priorP = np.dot(prior, cpMatrix)
                if (self.config.weightPoints):
                    jtv = np.dot(np.transpose(jac), np.dot(self.config.pointWeight, meanShiftVector))
                else:
                    jtv = np.dot(np.transpose(jac), meanShiftVector)
                paramUpdateLeft = np.add(prior, jtj)
                paramUpdateRight = np.subtract(priorP, jtv)
                paramUpdate = np.dot(np.linalg.inv(paramUpdateLeft), paramUpdateRight)
                oldPositions = deepcopy(self.currentPositions)
                self.update_timer('updating pdm parameter')

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
                self.update_timer('updating currentParameters')

                start = time.time()
                self.patchPositions, self.currentPositions = calculatePositions2(self.currentParameters, self.config.meanXShape, self.config.meanYShape,
                                                                                 self.config.xEigenVectors, self.config.yEigenVectors, True)
                self.update_timer('updating positions')

                if self.debug >= 2:
                    dst = gray.copy()
                    for x, y in self.currentPositions:
                        cv2.circle(dst, (int(x), int(y)), 2, (255))
                    plt.imshow(dst, cmap='gray')
                    plt.title('fine-grained face landmarks first iteration')
                    plt.show()

                start = time.time()
                positionNorm = np.sum(np.power(self.currentPositions - oldPositions, 2))
                logging.debug('iteration {}: position norm is {}, limit is {}'.format(iteration_minor, positionNorm, self.config.convergenceLimit))
                self.update_timer('calculating positionNorm')
                if positionNorm < self.config.convergenceLimit:
                    break

            self.previousPositions.append(self.currentPositions)
            if self.config.constantVelocity:
                self.previousParameters.append(self.currentParameters)
            # convergenceScore = getConvergence(self.previousPositions)
            # logging.debug('iteration {}: convergence score is {}'.format(iteration_minor, convergenceScore))
            # if convergenceScore < self.config.convergenceThreshold:
            #     logging.debug('CLM tracker converged: score is {} < {}'.format(convergenceScore, self.config.convergenceLimit))
            #     return True, face_score
            return True, face_score

def main():
    if len(sys.argv) < 2:
        logging.error('no model specified')
        sys.exit(1)
    tracker = clmFaceTracker(sys.argv[1], debug=False)

    imagePath = os.path.join(fileDir, '../examples/images/franck_02159.jpg')
    img = cv2.imread(imagePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (625, 500))
    gray=img[:,:,0]*0.3+img[:,:,1]*0.59+img[:,:,2]*0.11
    # the following conversion to integer pixel val makes a lot of arithmetic faster
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # plt.imshow(gray, cmap='gray')
    # plt.show()
    iteration = 0
    while True:
        converged, score = tracker.track(gray)
        iteration += 1
        print 'CLMTracker iteration {}: score: {}'.format(iteration, score)
        if converged:
            print 'CLMTracker converged after {} iterations'.format(iteration)
            break

    #    tracker.track

if __name__ == "__main__":
    main()
