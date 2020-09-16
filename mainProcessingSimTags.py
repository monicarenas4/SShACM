import numpy
import time
import datetime
from scoreAlign import alignScore
from scoreMaskedBlobs import maskedscore
from os import listdir
from os.path import join, isfile

colorthreshold = 5;
today = str(datetime.datetime.today())

refFolder = 'referenceImages'
tagFolder = 'simulatedTag'

results_Path = 'results'
type = 'SimulatedTags'
txtFile = results_Path + '/' + 'Results_' + type + '_' + today[:10] + '.txt'

referenceFolder = [f for f in listdir(refFolder) if isfile(join(refFolder, f))]
images = numpy.empty(len(referenceFolder), dtype=object)

for n in range(0, len(referenceFolder)):

    referenceImage = [f for f in listdir(tagFolder) if isfile(join(tagFolder, f))]
    images = numpy.empty(len(referenceImage), dtype=object)

    for m in range(0, len(referenceImage)):
        start = time.perf_counter()
        maskedscore(refFolder + '/' + referenceFolder[n], tagFolder + '/' + referenceImage[m])
        alignScore(refFolder + '/' + referenceFolder[n], tagFolder + '/' + referenceImage[m], colorthreshold)
        finish = time.perf_counter()
        print(m, referenceImage[m], f'{finish - start:0.2f}', f'{finish:0.2f}')

        with open(results_Path + '/' + 'Results_' + type + '_' + today[:10] + '.txt', 'a') as results_file:
            results_file.write(str(m + 1) + '\t' + f'{finish - start:0.2f}' + '\t')

print(time.perf_counter())
