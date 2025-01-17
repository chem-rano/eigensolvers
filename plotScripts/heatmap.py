from matplotlib import pyplot as plt
import numpy as np
import os
import seaborn as sns
import matplotlib.patches as patches

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
sqOvlp.append(np.genfromtxt(guessFile, skip_header=1, skip_footer=restrictRef,usecols=(3)))
cumIt.append(0)

file = path+"Overlap1.out"
refE = np.genfromtxt(file, skip_header=1, skip_footer=restrictRef,usecols=(1))
for i in range(1,maxCum+1): # must start from 1
    try:
        file = path+f"Overlap{i}.out"
        sqOvlp.append(np.genfromtxt(file, skip_header=1, skip_footer=restrictRef,usecols=(4)))
        cumIt.append(i)
    except FileNotFoundError:
        break
    except OSError:
        break

sqOvlp = np.array(sqOvlp)

# Plotting the heatmap
plt.figure(figsize=(12, 8))
ax = sns.heatmap(
    sqOvlp,
    xticklabels=np.round(refE, 1),
    yticklabels=cumIt,
    cmap="viridis",
    cbar_kws={'label': 'Squared Overlap'}
)

# Create rectangle
# ((x,y), width, height)
xRect = np.abs(refE-erange[0]).argmin()
yRect = plt.ylim()[1]+0.05
width = np.abs(refE-erange[1]).argmin() - xRect
height =  int(cumIt[-1])+.9
rect = patches.Rectangle((xRect,yRect),width,height,fill=False,linewidth=2,edgecolor='red',facecolor="None")
ax.add_patch(rect)

# Customize x-tick labels to skip some (e.g., show every other label)
xticks = ax.get_xticks()
xtick_labels = [label.get_text() if i % 2 == 0 else "" for i, label in enumerate(ax.get_xticklabels())]
ax.set_xticklabels(xtick_labels)

# Rotate and adjust x-tick labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
# Axis labels and title
plt.xlabel("Reference energies (D70)")
plt.ylabel("Cumulative Iterations")
plt.title(f"Heatmap of Squared Overlap: reference vs Iterations (Target {target}) DMRG-tol 1e-12")

plt.tight_layout()
plt.savefig(f"HeatmapRect.png")
