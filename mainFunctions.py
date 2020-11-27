from __future__ import print_function
import multiprocessing as mp
from multiprocessing import Pool
from numpy import power as pw
from os import listdir
from os.path import join, isfile
from skimage.feature import blob_log
from skimage.color import rgb2gray

import cv2
import datetime
import numpy as np
import re
import time
import matplotlib.pyplot as plt

today = str(datetime.datetime.today())
refPath = 'referenceImages'
results_Path = 'results'
type = 'retakenImages'
txtFile = results_Path + '/' + today[:10] + '_' + type + '.txt'

referenceFolder = [f for f in listdir(refPath) if isfile(join(refPath, f))]

MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.15


def alignImages(retake, reference):
    # imRetake = cv2.imread(retake, cv2.IMREAD_COLOR)  # Read reference image
    gray_retake = cv2.cvtColor(retake, cv2.COLOR_BGR2GRAY)
    # print(gray_retake.shape)

    # imReference = cv2.imread(reference, cv2.IMREAD_COLOR)  # Read image to be aligned
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    # print(gray_reference.shape)

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

    # Draw top matches
    # imMatches = cv2.drawMatches(imRetake, keypoints1, imReference, keypoints2, matches, None)
    # cv2.imwrite(name_ref + '_' + name_ret + '_matches.jpg', imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    height, width, channels = reference.shape
    imAlign = cv2.warpPerspective(retake, h, (width, height))
    # print(imAlign.shape)

    return imAlign, h


def alignScore(retake, reference, cp=5):
    imReference = cv2.imread(reference, cv2.IMREAD_COLOR)
    imRetaken = cv2.imread(retake, cv2.IMREAD_COLOR)
    imAligned, h = alignImages(imRetaken, imReference)

    imdiff = imAligned - imReference
    # plt.imshow(imdiff), plt.show()
    score1 = 1 - np.count_nonzero(imdiff) / imdiff.size
    score2 = (imdiff < cp).sum() / imdiff.size
    # print(score1, score2)

    # with open(txtFile, 'a') as results_file:
    #     results_file.write(str(score1) + '\n')

    return score1, score2


def maskedScore(retake, reference, threshold=3, corner=2):
    imgReference = cv2.imread(reference, cv2.IMREAD_COLOR)  # Read the reference image
    Ref_y, Ref_x, Ref_z = imgReference.shape  # Reference image shape
    gray_reference = rgb2gray(imgReference)  # Gray transformation
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

    # Aligned image
    imRetaken = cv2.imread(retake, cv2.IMREAD_COLOR)
    imAligned, h = alignImages(imRetaken, imgReference)
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

    print('I have found', m1, ' blobs for', reference, 'and ', m2, ' blobs for', retake)

    # remove the blobs near the image corner
    det_blobs_ref = (np.zeros([Ref_y, Ref_x, Ref_z])).astype(np.uint8)
    for i in range(0, m1):
        cv2.circle(det_blobs_ref, (b1[i][1], b1[i][0]), b1[i][2], [255, 255, 255], -5)
    # plt.imshow(det_blobs_ref), plt.show()
    # cv2.imwrite('detected_blobs_ref.jpg', det_blobs_ref)  # Write the detected blobs

    det_blobs_align = (np.zeros([imAlg_y, imAlg_x, imAlg_z])).astype(np.uint8)
    for i in range(0, m2):
        cv2.circle(det_blobs_align, (b2[i][1], b2[i][0]), b2[i][2], [255, 255, 255], -5)
    # plt.imshow(det_blobs_align), plt.show()

    count_detected, count_color, count_no_color = 0, 0, 0

    for i in range(0, m1):
        hide = []
        if m2 == 0:
            scorecolor = 0
        else:
            for j in range(0, m2):
                distance = np.sqrt(pw(b1[i][0] - b2[j][0], 2) + pw(b1[i][1] - b2[j][1], 2) + pw(b1[i][2] - b2[j][2], 2))
                hide.append(distance)

            if np.array(hide).min() < threshold:
                count_detected += 1
                c1_av1 = np.mean((imgReference[b1[i][0] - 2:b1[i][0] + 2, b1[i][1] - 2:b1[i][1] + 2]).reshape(16, 3),
                                 axis=0)
                indexmax1 = np.argmax(c1_av1)

                d1_av1 = np.mean((imAligned[b1[i][0] - 2:b1[i][0] + 2, b1[i][1] - 2:b1[i][1] + 2]).reshape(16, 3),
                                 axis=0)
                indexmax2 = np.argmax(d1_av1)

                if indexmax1 == indexmax2:
                    count_color += 1
                else:
                    count_no_color += 1

    scorecolor = (2 * count_color) / (m1 + m2)
    # print(scorecolor)

    # with open(txtFile, 'a') as results_file:
    #     name_ref = [m.start() for m in re.finditer(r"/", reference)][-1]
    #     name_ret = [m.start() for m in re.finditer(r"/", retake)][-1]
    #
    #     results_file.write(
    #         reference[name_ref + 1:] + '\t' +
    #         retake[name_ret + 1:] + '\t' +
    #         str(m1) + '\t' + str(m2) + '\t' +
    #         str(count_detected) + '\t' + str(scorecolor) + '\t')

    return (scorecolor)


for refFolder in referenceFolder:
    folderName = type + '/' + refFolder[:-4]
    retakeImages = [f for f in listdir(folderName) if isfile(join(folderName, f))]

    # for m in range(0, len(retakeImages)):
    for imageName in retakeImages:
        start = time.perf_counter()
        referenceImage = refFolder

        cores = (mp.cpu_count())
        with Pool(cores) as pool:
            pool.map(alignScore(folderName + '/' + imageName, folderName + '/' + referenceImage))
            pool.map(maskedScore(folderName + '/' + imageName, folderName + '/' + referenceImage))

        # pool.apply(maskedScore(folderName + '/' + retakeImages[m], folderName + '/' + referenceImage))
        # pool.apply(alignScore(folderName + '/' + retakeImages[m], folderName + '/' + referenceImage))
        finish = time.perf_counter()
        # pool.close()

        print(imageName, 'Sequential time:', f'{finish - start:0.2f}', 's')
