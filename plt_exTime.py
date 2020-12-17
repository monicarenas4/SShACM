import matplotlib.pyplot as plt
import pandas as pd

global_time = 'results/2020-12-15_globalFunction_Time_MPI.txt'
local_time = 'results/2020-12-15_localFunction_Time_MPI.txt'

df_glob = pd.read_csv(global_time, delimiter='\t')
df_loc = pd.read_csv(local_time, delimiter='\t')
cpu_glob = df_glob['cpus']
cpu_loc = df_loc['cpus']
time_glob = df_glob['time'] / 40
time_loc = df_loc['time']

plt.plot(cpu_glob, time_glob,
         linestyle='--', marker='o', color='tab:red',
         label='Global function')
plt.plot(cpu_loc, time_loc,
         linestyle='-.', marker='*', color='tab:blue',
         label='Local function')
plt.xlim([1, 8])
plt.grid()
plt.xlabel('Number of CPUs', fontsize=12)
plt.ylabel('Time (s)', fontsize=12)
plt.legend(fontsize=12)
plt.title('Execution time depending on the number of cores \n'
          'for a single image')
plt.show()
