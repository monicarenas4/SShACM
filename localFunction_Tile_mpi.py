from __future__ import print_function
from numpy import power as pw
from skimage.feature import blob_log
from skimage.color import rgb2gray

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import cv2
import datetime
import multiprocessing as mp
import numpy as np
import time

today = str(datetime.datetime.today())
results_Path = 'results/'
type = '_localFunction_'
txtFile = results_Path + today[:10] + type + 'Images_MPI.txt'

refImage = '/Image_228.jpg'
folderName = 'dataset' + refImage[:-4]
referenceImage = folderName + refImage
retakenImage = folderName + '/Image_228_blur_2_scaled_107.jpg'

img_ref = cv2.resize(cv2.imread(referenceImage),
                     (1600, 1200),
                     interpolation=cv2.INTER_AREA)

y_ref, x_ref, c_ref = img_ref.shape

img_ret = cv2.resize(cv2.imread(retakenImage),
                     (x_ref, y_ref),
                     interpolation=cv2.INTER_AREA)

cpus = range(1, mp.cpu_count() + 1)
# cpus = range(1, 6)

numrows, numcols = 2, 2
height = int(img_ref.shape[0] / numrows)
width = int(img_ref.shape[1] / numcols)

MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.15


def alignImages(retake, reference):
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    height, width, channels = reference.shape

    # imRetake = cv2.resize(retake, (width, height), interpolation=cv2.INTER_AREA)
    gray_retake = cv2.cvtColor(retake, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(gray_retake, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_reference, None)

    # Match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    matches.sort(key=lambda x: x.distance, reverse=False)  # Sort matches by score

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    imAlign = cv2.warpPerspective(retake, h, (width, height))

    return imAlign, h


def scoresFunction(retake, reference, cp=5, threshold=3, corner=2):
    ts1 = time.time()
    # imgReference = cv2.imread(reference, cv2.IMREAD_COLOR)
    Ref_y, Ref_x, Ref_z = reference.shape
    gray_reference = rgb2gray(reference)

    b1 = (blob_log(gray_reference,
                   min_sigma=10, max_sigma=30, num_sigma=10,
                   threshold=0.07, overlap=1)).astype(int)  # Blob's detection

    m, n = b1.shape
    pos = []
    for i in range(0, m):
        if ((b1[i][0] < corner) | (b1[i][1] < corner) | ((b1[i][0] + corner) > (Ref_y)) | (
                (b1[i][1] + corner) > (Ref_x))):
            pos.append(i)
    b1 = np.delete(b1, pos, axis=0)

    # imRetaken = cv2.imread(retake, cv2.IMREAD_COLOR)
    imAligned, h = alignImages(retake, reference)
    imAlg_y, imAlg_x, imAlg_z = imAligned.shape
    gray_align = rgb2gray(imAligned)
    b2 = (blob_log(gray_align,
                   min_sigma=10, max_sigma=30, num_sigma=10,
                   threshold=0.07, overlap=1)).astype(int)

    m, n = b2.shape
    pos = []
    for i in range(0, m):
        if ((b2[i][0] < corner) | (b2[i][1] < corner) | ((b2[i][0] + corner) > (Ref_y)) | (
                (b2[i][1] + corner) > (Ref_x))):
            pos.append(i)
    b2 = np.delete(b2, pos, axis=0)

    m1, n1 = b1.shape
    m2, n2 = b2.shape

    # remove the blobs near the image corner
    det_blobs_ref = (np.zeros([Ref_y, Ref_x, Ref_z])).astype(np.uint8)
    for i in range(0, m1):
        cv2.circle(det_blobs_ref, (b1[i][1], b1[i][0]), b1[i][2], [255, 255, 255], -5)

    det_blobs_align = (np.zeros([imAlg_y, imAlg_x, imAlg_z])).astype(np.uint8)
    for i in range(0, m2):
        cv2.circle(det_blobs_align, (b2[i][1], b2[i][0]), b2[i][2], [255, 255, 255], -5)
    # plt.imshow(det_blobs_align), plt.show()

    # # # # # # Blob Scores # # # # # #
    count_detected, count_color, count_no_color = 0, 0, 0

    for i in range(0, m1):
        hide = []
        if (m1 == 0) or (m2 == 0):
            scorecolor = 0
        else:
            for j in range(0, m2):
                distance = np.sqrt(pw(b1[i][0] - b2[j][0], 2) + pw(b1[i][1] - b2[j][1], 2) + pw(b1[i][2] - b2[j][2], 2))
                hide.append(distance)

            if np.array(hide).min() < threshold:
                count_detected += 1
                c1_av1 = np.mean((reference[b1[i][0] - 2:b1[i][0] + 2, b1[i][1] - 2:b1[i][1] + 2]).reshape(16, 3),
                                 axis=0)
                indexmax1 = np.argmax(c1_av1)

                d1_av1 = np.mean((imAligned[b1[i][0] - 2:b1[i][0] + 2, b1[i][1] - 2:b1[i][1] + 2]).reshape(16, 3),
                                 axis=0)
                indexmax2 = np.argmax(d1_av1)

                if indexmax1 == indexmax2:
                    count_color += 1
                else:
                    count_no_color += 1

    if (m1 + m2) != 0:
        scorecolor = (2 * count_color) / (m1 + m2)
    else:
        scorecolor = 0

    # # # # # # Alignment Scores # # # # # #
    imdiff = imAligned - reference
    score1 = 1 - np.count_nonzero(imdiff) / imdiff.size
    score2 = (imdiff < cp).sum() / imdiff.size

    with open(txtFile, 'a') as results_file:

        results_file.write(
            f'{time.time() - ts1}' + '\t' +
            str(m1) + '\t' +
            str(m2) + '\t' +
            str(count_detected) + '\t' +
            str(scorecolor) + '\t' +
            str(score1) + '\t' +
            str(score2) + '\n')

    return (score1, score2, scorecolor)


def result_alignScore(score1):
    global align
    align.append(score1)


if __name__ == '__main__':
    ts1 = time.time()
    align = []
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    num_workers = max(size, size - 1)
    executor = MPIPoolExecutor(num_workers)
    print(str(rank) +'/'+ str(size))
    
    for cpu in cpus:
    
        for row in range(numrows):
            ts2 = time.time()
            for col in range(numcols):
                y0 = row * height
                y1 = y0 + height
                x0 = col * width
                x1 = x0 + width
                ref = img_ref[y0:y1, x0:x1]
                ret = img_ret[y0:y1, x0:x1]

                # cv2.imwrite('tile/tileRef_%d%d.jpg' % (row, col), ref)
                # cv2.imwrite('tile/tileRet_%d%d.jpg' % (row, col), ret)
                future = executor.submit(scoresFunction,[ret,ref], result_alignScore)

        parFile = results_Path + today[:10] + type + 'Time_MPI.txt'
        with open(parFile, 'a') as par_file:

            par_file.write(
                str(cpu) + '\t' +
                str(round((time.time() - ts2), 2)) + '\n')

        print((time.time() - ts2), 'seconds')
