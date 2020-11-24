1. image_1104.jpg
1. image_2366.jpg
********

# Alignment Implementation
This implementation is based on image processing algorithms for:
- Estimating the alignment between retake and reference images: `align(imRetake, imReference)`;
- Computing the scores related to the blobs position and color detection. 


## Reference and mutated images
The [reference images](https://github.com/monicarenas4/SShParallelComputing/tree/master/referenceImages)
are composed by colored blobs distributed randomly.
In order to simulate [retaken images](https://github.com/monicarenas4/SShParallelComputing/tree/master/mutatedImages) 
some noise was introduced to the reference 
images, named as  _mutated images_. Thus, each _reference images_ was resized, 
rotated, cropped, and blurred.


## Problems
Some images presented problems in the alignment procedure, for this 
reason different approaches have been proposed.


## Alignment
The Homography matrix (3×3) is at the heart of image alignment techniques.
A Homography matrix is a transformation that maps the keypoints
of one image to the corresponding points in the other image.
The homography can be calculated _only if_ it is known the corresponding features of the two images.
To calculate the homography between two images, it is required to know at least
4 keypoint correspondences between the images (bigger correspondence are preferred).

The keypoint used in this approach was the ORB (using the `opencv` library).

**Key concepts:**
- ORB stands for _Oriented FAST and Rotated BRIEF_.
- **FAST** localizes interesting points but does not deal with the identity of the point.
- **BRIEF** describes the region around the point and it can be identified in a different image.

The recognition problem should be solved in two steps: 
1. Localization: detector that output the coordinates of the located blobs. 
1. Recognition.

# Alignment Implementation
This implementation is based on image processing algorithms for:
- Estimating the alignment between retake and reference images: `align(imRetake, imReference)`;
- Computing the scores related to the blobs position and color detection. 


## Reference and mutated images
The [reference images](https://github.com/monicarenas4/SShParallelComputing/tree/master/referenceImages)
are composed by colored blobs distributed randomly.
In order to simulate [retaken images](https://github.com/monicarenas4/SShParallelComputing/tree/master/mutatedImages) 
some noise was introduced to the reference 
images, named as  _mutated images_. Thus, each _reference images_ was resized, 
rotated, cropped, and blurred.


## Problems
Some images presented problems in the alignment procedure, for this 
reason different approaches have been proposed.


## Alignment
The Homography matrix (3×3) is at the heart of image alignment techniques.
A Homography matrix is a transformation that maps the keypoints
of one image to the corresponding points in the other image.
The homography can be calculated _only if_ it is known the corresponding features of the two images.
To calculate the homography between two images, it is required to know at least
4 keypoint correspondences between the images (bigger correspondence are preferred).

The keypoint used in this approach was the ORB (using the `opencv` library).

**Key concepts:**
- ORB stands for _Oriented FAST and Rotated BRIEF_.
- **FAST** localizes interesting points but does not deal with the identity of the point.
- **BRIEF** describes the region around the point and it can be identified in a different image.

The recognition problem should be solved in two steps: 
1. Localization: detector that output the coordinates of the located blobs. 
1. Recognition.

The steps are indicated in the following diagram.
![alt text](https://github.com/monicarenas4/SShParallelComputing/blob/master/methodology.jpg)

****
## Alignment score
This score is based on the difference between the _aligned image_ and the _reference
image_. If the reference and retake image are the same, the _alignment score_ is 1. 
If both images are completely different or the alignment fails, this score is 0.

```python
imdiff = imAligned - imReference
alignScore = 1 - nonzero(imdiff) / size(imdiff)
``` 

### Counting blobs and color detection
- The reference image is transformed in gray scale. 
    - The number of blobs are detected. 
    - The blobs located at the corners are eliminated (∓ 2 pixel). 
- Alignment between the reference and the retake images: `align(imRetake, imReference)`
- The aligned image is transformed into gray scale
    - The number of blobs are detected. 
    - The blobs located at the image corner are removed (∓ 2 pixel). 
- The Euclidean distance between the detected blobs (from the aligned and reference images) is computed. 
If the Euclidean distance is lower than a set threshold, then the blob is  counted.
```python
if {min(euclideanDistance) < threshold} => blob_detected
```
- The score color is computed as: 
```python
scorecolor = (2 * count_color) / (m1 + m2)
```

```python
imdiff = imAligned - imReference
alignScore = 1 - nonzero(imdiff) / size(imdiff)
``` 

### Counting blobs and color detection
- The reference image is transformed in gray scale. 
    - The number of blobs are detected. 
    - The blobs located at the corners are eliminated (∓ 2 pixel). 
- Alignment between the reference and the retake images: `align(imRetake, imReference)`
- The aligned image is transformed into gray scale
    - The number of blobs are detected. 
    - The blobs located at the image corner are removed (∓ 2 pixel). 
- The Euclidean distance between the detected blobs (from the aligned and reference images) is computed. 
If the Euclidean distance is lower than a set threshold, then the blob is  counted.
```python
if {min(euclideanDistance) < threshold} => blob_detected
```
- The score color is computed as: 
```python
scorecolor = (2 * count_color) / (m1 + m2)
```