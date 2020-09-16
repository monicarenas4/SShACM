**[Reference images](https://github.com/monicarenas4/SShACM/tree/master/referenceImages):** coming from 10 different droplet tags.
1. image_228.jpg
1. image_618.jpg
1. image_852.jpg
1. image_972.jpg
1. image_974.jpg
1. image_975.jpg
1. image_997.jpg
1. image_1060.jpg
1. image_1079.jpg
1. image_1103.jpg
1. image_1104.jpg
1. image_2366.jpg
********
### Experiments, Scores, and Evaluation
From each *reference image* were generated around 200 mutated images (folder: **mutatedImages**).
 Example of [mutated images](https://github.com/monicarenas4/SShImageProcessing/tree/master/mutatedImages/Image_1104):
![alt text](https://github.com/monicarenas4/SShACM/blob/master/RefMut.png)
We generated a set of 1000 simulated tags, listed in *simulatedTag* folder. 

### Validation
For the validations of the proposed methodology, from each reference image were generated around 50 mutated images,
as observed in the folder *mutatedImagesTesting*. We also generated a set of simulated tags (100 images), 
as listed in the folder *simulatedTagTesting*.

****

The **mainProcessing.py** file is used for running the experiments related to the *mutated images*
and the file **mainProcessingSimTags.py** is used for running the experiments related to the the 
simulated tags. **IMPORTANT:** depending on the kind of experiments that it is wished to run, it is required
to change the directory path in **scoreAlign.py** and **scoreMaskedBlobs.py** files.
