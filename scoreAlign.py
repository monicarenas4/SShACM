from __future__ import print_function
from alignImages import alignImages
import cv2
import datetime
import time
import numpy as np

today = str(datetime.datetime.today())
results_Path = 'results'
type = 'retakenImages'
txtFile = results_Path + '/' + today[:10] + '_' + type + '.txt'


def alignScore(f1, f2, cp):
    start = time.perf_counter()
    imReference = cv2.imread(f1, cv2.IMREAD_COLOR)  # Read reference image
    imAligned = cv2.imread(f2, cv2.IMREAD_COLOR)  # Read image to be aligned
    imReg, h = alignImages(imAligned, imReference)

    imdiff = imReg - imReference
    score1 = 1 - np.count_nonzero(imdiff) / imdiff.size
    score2 = (imdiff < cp).sum() / imdiff.size

    end = time.perf_counter()

    with open(txtFile, 'a') as results_file:
        # results_file.write(str(score1 * 100) + '\t' + str(score2 * 100) + '\n')
        results_file.write(str(score1) + '\n')

    return score1, score2
