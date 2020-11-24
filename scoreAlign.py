from __future__ import print_function
from alignImages import alignImages
import cv2
import datetime
import time
import matplotlib.pyplot as plt
import numpy as np

today = str(datetime.datetime.today())
results_Path = 'results'
type = 'retakenImages'
txtFile = results_Path + '/' + today[:10] + '_' + type + '.txt'


def alignScore(retake, reference, cp):
    imReference = cv2.imread(reference, cv2.IMREAD_COLOR)
    imRetaken = cv2.imread(retake, cv2.IMREAD_COLOR)
    imAligned, h = alignImages(imRetaken, imReference)

    imdiff = imAligned - imReference
    # plt.imshow(imdiff), plt.show()
    score1 = 1 - np.count_nonzero(imdiff) / imdiff.size
    score2 = (imdiff < cp).sum() / imdiff.size
    print(score1, score2)

    with open(txtFile, 'a') as results_file:
        results_file.write(str(score1) + '\n')

    return score1, score2
