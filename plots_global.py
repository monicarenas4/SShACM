from itertools import cycle
from numpy import cumsum
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cores = 8
setValue = 26
# rng = ['40', '80', '120', '160', '200', '240', '280', '320']
rng = [40, 80, 120, 160, 200, 240, 280, 320]
lines = ["-", "--", "-.", ":"]
linecycler = cycle(lines)

file_complex = 'results/2020-12-17_globalFunction_Images_MPI.txt'
df_complex = pd.read_csv(file_complex, delimiter='\t')
time_per_image = (df_complex['time']).to_numpy()
val1 = round((cumsum(time_per_image[:setValue]))[-1], 2)
val2 = round((cumsum(time_per_image[280:304]))[-1] / cores, 2)

a = 0
for ln in range(len(rng)):
    cpu_time = (cumsum(time_per_image[a:rng[ln]]) / (ln + 1))
    plt.plot(cpu_time, next(linecycler), label='CPUs = ' + str(ln + 1))
    a = rng[ln]

plt.axvline(x=setValue, color='red', linestyle='-.')
plt.xlim([0, 39])
plt.ylim([0, 850])
plt.annotate(str(val1), xy=(setValue, val1))  # , xytext=(setValue, 530))
plt.annotate(str(val2), xy=(setValue, val2), xytext=(setValue, 75))
plt.grid()
# plt.xticks(np.arange(1, 41, 5))
plt.xticks(np.arange(1, 41, 5))
plt.xlabel('Problem size', fontsize=12)
plt.ylabel('Time (s)', fontsize=12)
plt.legend(fontsize=12)
plt.title('Complexity')
plt.show()

file_time = 'results/2020-12-17_globalFunction_Time_MPI.txt'

df_par = pd.read_csv(file_time, delimiter='\t')
cpu = df_par['cpus']
time_per_bloc = df_par['time']

plt.plot(cpu, time_per_bloc,
         linestyle='--', marker='o', color='tab:red',
         label='Problem size = 40')
plt.xlim([1, 8])
plt.grid()
plt.xlabel('Number of CPUs', fontsize=12)
plt.ylabel('Time (s)', fontsize=12)
plt.legend(fontsize=12)
plt.title('Execution time depending on the number of cores using MPI')
plt.show()
