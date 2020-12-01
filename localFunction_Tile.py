from __future__ import print_function
import multiprocessing as mp
from math import ceil
from numpy import power as pw
from skimage.feature import blob_log
from skimage.color import rgb2gray

import cv2
import datetime
import numpy as np
import re
import time
import matplotlib.pyplot as plt

today = str(datetime.datetime.today())
results_Path = 'results'
type = 'retakenImages'
testImage = "Image_228"

folderPath = type + '/' + testImage
referenceImage = folderPath + "/" + testImage + ".jpg"
retakenImage = folderPath + "/" + "Image_228_scaled_94_crop_3_8_rot_354_blur_1_1.jpg"

txtFile = results_Path + '/' + today[:10] + '_localFunction' + '.txt'

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
        if m2 == 0:
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

        # name_ref = [m.start() for m in re.finditer(r"/", reference)][-1]
        # name_ret = [m.start() for m in re.finditer(r"/", retake)][-1]

        results_file.write(
            # reference[name_ref + 1:] + '\t' +
            # retake[name_ret + 1:] + '\t' +
            str(m1) + '\t' + str(m2) + '\t' +
            str(count_detected) + '\t' +
            str(scorecolor) + '\t' +
            str(score1) + '\t' +
            str(score2) + '\t' +
            # +   )  +
            f'{time.time() - ts1}' + '\n')

    # print(time.time() - ts1)

    return (score1, score2, scorecolor)


img_ref = cv2.imread(retakenImage)
img_ref = cv2.resize(img_ref, (512, 512), interpolation=cv2.INTER_CUBIC)
y_ref, x_ref, c_ref = img_ref.shape

img_ret = cv2.imread(retakenImage)
img_ret = cv2.resize(img_ret, (512, 512), interpolation=cv2.INTER_CUBIC)
y_ret, x_ret, c_ret = img_ret.shape

tile_size = (256, 256)
offset = (256, 256)

ts = time.time()
for i in range(ceil(x_ref / (offset[1] * 1.0))):
    for j in range(ceil(y_ref / (offset[0] * 1.0))):
        tile_ref = img_ref[offset[1] * i:min(offset[1] * i + tile_size[1], y_ref),
                   offset[0] * j:min(offset[0] * j + tile_size[0], x_ref)]
        tile_ret = img_ret[offset[1] * i:min(offset[1] * i + tile_size[1], y_ret),
                   offset[0] * j:min(offset[0] * j + tile_size[0], x_ret)]
        scoresFunction(tile_ret, tile_ref)
        # Debugging the tiles
        # plt.imshow(tile_ref), plt.show()
        # plt.imshow(tile_ret), plt.show()

print(f'{time.time() - ts}')