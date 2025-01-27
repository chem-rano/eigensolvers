import numpy as np
from ttns2.state import loadTTNSFromHdf5
import util
import warnings
from ttns2.driver import bracket
import os

# *****************  Description ****************************
# This script checks state correspondence after refinement
# as per the overlap criteria
# ***********************************************************

zpve = 9837.4069
# ...................... Root file ..............
prefix = "/home/madhumitarano/data/PR39/ch3cn/250states/"
files = np.loadtxt("../../../batchRoots.dat",skiprows=0,usecols=(1),dtype=str)

if files.ndim == 0:
    files = files.reshape(1)
batchsize = len(files)

rootStates = []
rootEnergies = []
for i in range(batchsize):
    filename = prefix + files[i]
    rootStates.append(loadTTNSFromHdf5(filename)[0])
    rootEnergies.append(loadTTNSFromHdf5(filename)[1]["energy"])
rootEnergies = util.au2unit(np.array(rootEnergies),"cm-1")-zpve
# -------------------------------------------------------
overlap_file = open(f"RefinedStateCorrespondence.out","w")
formatStyle = "{:>10} {:>20} {:>14} {:>18} {:>12}"
lines = formatStyle.format("root-Index","refinedState-Index","energy-root","energy-refinedState","overlap")+"\n"
overlap_file.write(lines)
final_refinedStates = []
final_refinedEnergies = []

for i in range(batchsize):
    maxOvlp = 0.0
    maxOvlp_idx = None
    maxOvlp_stateE = np.inf

    rootState = rootStates[i]
    for idx in range(1000): 
        try:
            filename = f"../finalLanczosTNSs/lanczosSolution{idx}.h5"
            refinedState = loadTTNSFromHdf5(filename)[0]
            refinedStateEnergy = util.au2unit(loadTTNSFromHdf5(filename)[1]["energy"],"cm-1")-zpve
            overlap = bracket(rootState,refinedState)
            
            # if refined state is in the list, skip
            if idx in final_refinedStates:
                continue

            if abs(overlap) > maxOvlp:
                maxOvlp = abs(overlap)
                maxOvlp_idx = idx
                maxOvlp_stateE = refinedStateEnergy
   
        except FileNotFoundError:
            break

    final_refinedStates.append(maxOvlp_idx)
    final_refinedEnergies.append(maxOvlp_stateE)
    lines = formatStyle.format(i,maxOvlp_idx,f"{rootEnergies[i]:.6f}",f"{maxOvlp_stateE:.6f}",f"{maxOvlp:.4f}")+"\n"
    overlap_file.write(lines)

final_refinedEnergies, final_refinedStates= zip(*sorted(zip(final_refinedEnergies, final_refinedStates)))
overlap_file.write(f"\n\nSorted refined-states energies: ")
overlap_file.write(f"{final_refinedEnergies}\n")
overlap_file.write(f"Sorted refined-states indices: ")
overlap_file.write(f"{final_refinedStates}\n")

os.chdir("../")
cwd = os.getcwd()
for idx in final_refinedStates:
    filename = f"{cwd}/finalLanczosTNSs/lanczosSolution{idx}.h5"
    overlap_file.write(filename+"\n")
overlap_file.close()
# -------------------------------------------------------
