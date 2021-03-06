from __future__ import print_function
from numpy import power as pw
from os import listdir
from os.path import join, isfile
from skimage.feature import blob_log
from skimage.color import rgb2gray

import cv2
import datetime
import multiprocessing as mp
import numpy as np
import re
import time

cpus = range(1, mp.cpu_count() + 1)
# cpus = range(1, 2)
N_SAMPLES = 100
MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.15

numbers = re.compile(r'(\d+)')


def numerical_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts



ref_image = '/Image_228.jpg'
folder_name = 'dataset' + ref_image[:-4]
reference_image = folder_name + ref_image

retake_list = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
retake_images = sorted(retake_list, key=numerical_sort)

today = str(datetime.datetime.today())
ref_path = 'referenceImages'
results_path = 'results/'
type = '_globalFunction_'
txt_file = results_path + today[:10] + type + 'Images.txt'

with open(txt_file, 'a') as par_file:
    par_file.write('reference' + '\t'
                   + 'retake' + '\t'
                   + 'time' + '\t'
                   + 'm1' + '\t'
                   + 'm2' + '\t'
                   + 'count_detected' + '\t'
                   + 'scorecolor' + '\t'
                   + 'score1' + '\t'
                   + 'score2' + '\n')


def align_images(retake, reference):
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    HEIGHT, WIDTH, CHANNELS = reference.shape

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

    imAlign = cv2.warpPerspective(retake, h, (WIDTH, HEIGHT))

    return imAlign, h


def scores_function(retake, reference, cp=5, threshold=3, corner=2):
    ts = time.time()

    imgReference = cv2.imread(reference, cv2.IMREAD_COLOR)
    Ref_y, Ref_x, Ref_z = imgReference.shape
    gray_reference = rgb2gray(imgReference)

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

    imRetaken = cv2.imread(retake, cv2.IMREAD_COLOR)
    imAligned, h = align_images(imRetaken, imgReference)
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

    # # # # # # Alignment Scores # # # # # #
    imdiff = imAligned - imgReference
    score1 = 1 - np.count_nonzero(imdiff) / imdiff.size
    score2 = (imdiff < cp).sum() / imdiff.size

    with open(txt_file, 'a') as results_file:

        name_ref = [m.start() for m in re.finditer(r"/", reference)][-1]
        name_ret = [m.start() for m in re.finditer(r"/", retake)][-1]

        results_file.write(reference[name_ref + 1:] + '\t'
                           + retake[name_ret + 1:] + '\t'
                           + str(round(time.time() - ts, 2)) + '\t'
                           + str(m1) + '\t' + str(m2) + '\t'
                           + str(count_detected) + '\t'
                           + str(scorecolor) + '\t'
                           + str(score1) + '\t'
                           + str(score2) + '\n')

    return (score1, score2, scorecolor)


def result_alignScore(score1):
    global align
    align.append(score1)


if __name__ == '__main__':
    ts1 = time.time()
    align = []

    for cpu in cpus:
        pool = mp.Pool(cpu)

        for retake in retake_images[:N_SAMPLES]:
            ts2 = time.time()
            # print(folderName + '/' + imageName + " test " + referenceImage)
            pool.apply_async(scores_function, args=(folder_name + '/' + retake,
                                                    reference_image),
                             callback=result_alignScore)

        pool.close()
        pool.join()

        parFile = results_path + today[:10] + type + 'Time.txt'

        with open(parFile, 'a') as par_file:
            par_file.write(
                str(N_SAMPLES) + '\t' +
                str(cpu) + '\t' +
                str(round((time.time() - ts2), 2)) + '\n')

        print((time.time() - ts2) / 60, 'minutes')
