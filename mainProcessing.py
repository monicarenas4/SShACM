import time
import datetime
from scoreAlign import alignScore
from scoreMaskedBlobs import maskedscore
from os import listdir
from os.path import join, isfile

colorthreshold = 5;
today = str(datetime.datetime.today())

refPath = 'referenceImages'
results_Path = 'results'
type = 'retakenImages'
txtFile = results_Path + '/' + today[:10] + '_' + type + '.txt'

referenceFolder = [f for f in listdir(refPath) if isfile(join(refPath, f))]

for refFolder in referenceFolder:
    folderName = type + '/' + refFolder[:-4]
    retakeImages = [f for f in listdir(folderName) if isfile(join(folderName, f))]

    for m in range(0, len(retakeImages)):
        start = time.perf_counter()
        challengeIm = refFolder
        maskedscore(folderName + '/' + challengeIm, folderName + '/' + retakeImages[m])
        alignScore(folderName + '/' + challengeIm, folderName + '/' + retakeImages[m], colorthreshold)
        finish = time.perf_counter()
        print(retakeImages[m], f'{finish - start:0.2f}', f'{finish:0.2f}')

        # with open(txtFile, 'a') as results_file:
        #     results_file.write(str(m + 1) + '\t' + f'{finish - start:0.2f}' + '\t')

# print(time.perf_counter())
