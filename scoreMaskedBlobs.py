import cv2
import numpy as np
from numpy import power as pw
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
    print(scorecolor)

    with open(txtFile, 'a') as results_file:
        results_file.write(
            reference[reference.find('/') + 1:] + '\t' + retake[retake.find('/') + 1:] + '\t' + str(m1) + '\t' + str(
                m2) + '\t' + str(
                count_detected) + '\t' + str(scorecolor) + '\t')

    return (scorecolor)
