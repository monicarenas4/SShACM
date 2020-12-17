from itertools import cycle
from numpy import cumsum
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cores = 8
setValue = 3
rng1 = ['1', '2', '3', '4']
rng = range(4, 34, 4)
lines = ["-", "--", "-.", ":"]
linecycler = cycle(lines)

file_complex = 'results/2020-12-16_localFunction_Images_MPI.txt'
df_complex = pd.read_csv(file_complex, delimiter='\t')
time_per_image = (df_complex['time']).to_numpy()
val1 = round((cumsum(time_per_image[:setValue+1]))[-1], 2)
val2 = round((cumsum(time_per_image[28:32]))[-1] / cores, 2)

a = 0
for ln in range(len(rng)):
    cpu_time = (cumsum(time_per_image[a:rng[ln]]) / (ln + 1))
    plt.plot(cpu_time, next(linecycler), label='CPUs = ' + str(ln + 1))
    a = rng[ln]

plt.axvline(x=setValue, color='red', linestyle='-.')
plt.xlim([0, 3])
plt.ylim([0, 28])
plt.annotate(str(val1), xy=(setValue, val1))  # , xytext=(setValue, 530))
plt.annotate(str(val2), xy=(setValue, val2)) #, xytext=(setValue, 75))
plt.grid()
plt.xticks(range(len(rng1)), rng1)
plt.xlabel('Problem size (# tile images)', fontsize=12)
plt.ylabel('Time (s)', fontsize=12)
plt.legend(fontsize=12)
plt.title('Time Complexity for the local function running with MPI')
plt.show()
