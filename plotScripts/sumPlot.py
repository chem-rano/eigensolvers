from matplotlib import pyplot as plt
import numpy as np
import os

target = -3306
erange = np.genfromtxt("demo/erange.dat",skip_header=0,skip_footer=0)
maxCum = 1000
path = "demo/"
restrictRef = 1378-50

#color: darkred,black
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,}

refE = []
sqOvlp = []
cumIt = []

guessFile = path + "guess.out"
sqOvlp_all = np.genfromtxt(guessFile, skip_header=1, skip_footer=0,usecols=(3))
refE_all = np.genfromtxt(guessFile, skip_header=1, skip_footer=0,usecols=(1))

indices = (refE_all >= erange[0]) & (refE_all <= erange[1])
refE.append(refE_all[indices])
sqOvlp.append(sum(sqOvlp_all[indices]))
cumIt.append(0)

file = path+"Overlap1.out"

for i in range(1,maxCum+1): # must start from 1
    try:
        file = path+f"Overlap{i}.out"
        sqOvlp_all =np.genfromtxt(file, skip_header=1, skip_footer=0,usecols=(4))
        sqOvlp.append(sum(sqOvlp_all[indices]))
        cumIt.append(i)
    except FileNotFoundError:
        break
    except OSError:
        break



plt.figure()
ax = plt.subplot(111)
ax.plot(cumIt,sqOvlp,lw=2.5,color="magenta")

plt.xlabel("Cumulative iterations",fontdict=font)
plt.ylabel("Sum of squared overlap",fontdict=font)
plt.title(f"Reference range: [{erange[0]},{erange[1]}] in {round(abs(erange[0]-erange[1]),2)} cm-1 interval")
plt.savefig(f"SumPlot.png")
