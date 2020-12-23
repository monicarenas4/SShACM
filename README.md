# <font color="darkblue"> Reference and retaken images </font>
Reference images are composed by colored blobs distributed randomly.
In order to simulate retaken images
some noise was introduced to the reference 
images. Thus, each _reference images_ was resized, cropped, and blurred.


## Alignment Implementation
This implementation is based on image processing algorithms for:
- Estimating the alignment score between retake and reference images:
  `align(retake, reference)`

# <font color="#00008b"> Problems </font> 
## Global parallelization
Retaken images (`retake_i`) are compared with a reference image (`reference`) 
in order to measure if the retaken images are authentic.
In the current implementation, the __sequential execution time__ for analyzing each
pair of images takes around `22 s`. When analyzing `1000` pair of images the total
execution time is approx. 7h.
However, it is desired to parallelize this process to reduce the computational time.
One proposal is to split the set of retaken images in different dataset and run in parallel
in order to optimize the total execution time.

## Local paralellization
It is also important to parallelize the local function `align(retake, reference)`
in order to reduce the 
computational time for matching each pair of images;
that as stated above, each process takes approx. `22 s`.
Thus, it is also proposed to parallelize the process of matching by checking
if all tiles do match: `align(retake, reference) = align_tile[g_i(retake),g_i(reference)]`.

