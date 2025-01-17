from matplotlib import pyplot as plt
import numpy as np
import os

target = -3306
file = "demo/summary_lanczos.out"

#color: darkred,black
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,}

it= np.genfromtxt(file, skip_header=26, skip_footer=0,usecols=(2))
eigenvalue = np.genfromtxt(file, skip_header=26, skip_footer=0,usecols=(4))

directory = "Energy/"
if not os.path.exists(directory):
    os.makedirs(directory)

plt.figure()
ax = plt.subplot(111)
ax.plot(it,eigenvalue,lw=2.5,label="Eigenvalue")

xlimits = ax.get_xlim()
ax.plot(xlimits,[target,target], marker=" ", linestyle ="dotted",lw=2.5, color="red",label="Target")
ax.legend()

plt.xlabel("Cumulative iteration",fontdict=font)
plt.ylabel("Eigenvalue (cm-1)",fontdict=font)
plt.savefig(directory+"energy.png")
