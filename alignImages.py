from __future__ import print_function
import cv2
import numpy as np

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
