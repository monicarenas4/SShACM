import cv2
import numpy as np
import datetime

from alignImages import alignImages
from skimage.feature import blob_log
from skimage.color import rgb2gray

MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.15
today = str(datetime.datetime.today())

results_Path = 'results'
type = 'retakenImages'
txtFile = results_Path + '/' + today[:10] + '_' + type + '.txt'


def maskedscore(f1, f2):
    threshold = 3;
    imgReference = cv2.imread(f1);
    img2 = cv2.imread(f2);

    imaligned, h = alignImages(img2, imgReference);

    gray1 = rgb2gray(imgReference)
    blobs_log1 = blob_log(gray1, min_sigma=10, max_sigma=30, num_sigma=10, threshold=0.07, overlap=1)
    b1 = blobs_log1.astype(int);

    imy1, imx1 = imgReference.shape[0], imgReference.shape[1];

    m, n = b1.shape
    pos = [];
    for i in range(0, m):
        if ((b1[i][0] < 2) | (b1[i][1] < 2) | ((b1[i][0] + 2) > (imy1)) | ((b1[i][1] + 2) > (imx1))):
            pos.append(i);
    b1 = np.delete(b1, pos, axis=0)

    gray2 = rgb2gray(imaligned)
    blobs_log2 = blob_log(gray2, min_sigma=10, max_sigma=30, num_sigma=10, threshold=0.07, overlap=1)
    b2 = blobs_log2.astype(int);

    m, n = b2.shape
    pos = [];
    for i in range(0, m):
        if ((b2[i][0] < 2) | (b2[i][1] < 2) | ((b2[i][0] + 2) > (imy1)) | ((b2[i][1] + 2) > (imx1))):
            pos.append(i);
    b2 = np.delete(b2, pos, axis=0)

    m1, n1 = b1.shape
    m2, n2 = b2.shape

    imgtry1 = np.zeros([imy1, imx1, 3])
    imgtry1 = imgtry1.astype(np.uint8)
    for i in range(0, m1):
        cv2.circle(imgtry1, (b1[i][1], b1[i][0]), b1[i][2], [255, 255, 255], -5)

    imy2, imx2 = imaligned.shape[0], imaligned.shape[1];
    imgtry2 = np.zeros([imy2, imx2, 3])
    imgtry2 = imgtry2.astype(np.uint8)
    for i in range(0, m2):
        cv2.circle(imgtry2, (b2[i][1], b2[i][0]), b2[i][2], [255, 255, 255], -5)

    imblobsonly1 = cv2.bitwise_and(imgReference, imgtry1);
    imblobsonly2 = cv2.bitwise_and(imaligned, imgtry2);  # Appply mask to get the blobs only

    countdetected = 0;
    countcolor = 0;
    countnocolor = 0;

    for i in range(0, m1):
        hide = []
        if m2 == 0:
            scorecolor = 0;
        else:
            for j in range(0, m2):
                distance = ((b1[i][0] - b2[j][0]) * (b1[i][0] - b2[j][0]) + (b1[i][1] - b2[j][1]) * (
                        b1[i][1] - b2[j][1]) + (b1[i][2] - b2[j][2]) * (b1[i][2] - b2[j][2]));
                hide.append(distance)

            mm = np.array(hide).min()
            if mm < threshold:
                countdetected = countdetected + 1;
                c = imgReference[b1[i][0] - 2:b1[i][0] + 2, b1[i][1] - 2:b1[i][1] + 2]
                c1 = c.reshape(16, 3)
                c1_av1 = np.mean(c1, axis=0)
                indexmax1 = np.argmax(c1_av1)

                d = imaligned[b1[i][0] - 2:b1[i][0] + 2, b1[i][1] - 2:b1[i][1] + 2]
                d1 = d.reshape(16, 3)
                d1_av1 = np.mean(d1, axis=0)
                indexmax2 = np.argmax(d1_av1)
                if indexmax1 == indexmax2:
                    countcolor = countcolor + 1;
                else:
                    countnocolor = countnocolor + 1;

    scorecolor = 2 * countcolor / (m1 + m2);
    with open(txtFile, 'a') as results_file:
        results_file.write(
            f1[f1.find('/') + 1:] + '\t' + f2[f2.find('/') + 1:] + '\t' + str(m1) + '\t' + str(m2) + '\t' + str(
                countdetected) + '\t' + str(scorecolor) + '\t')

    return (scorecolor)
