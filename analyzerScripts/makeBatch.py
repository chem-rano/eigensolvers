import time
import subprocess
import os
import glob
import numpy as np
import util

# ...................... Sorted reference file ..............
filename = "demo/Sorted_states.out"
skiprows = 1
roots = np.loadtxt(filename,skiprows=skiprows,usecols=(0),dtype=int)
energies = np.loadtxt(filename,skiprows=skiprows,usecols=(2),dtype=float)
files = np.loadtxt(filename,skiprows=skiprows,usecols=(4),dtype=str)

energies = list(energies) # okay to convert
def group_by_interval(arr, interval):
    if not arr:
        return []

    # Sort the array to make grouping easier
    arr.sort()

    groups = []
    current_group = [arr[0]]

    for i in range(1, len(arr)):
        # Check if the current element fits in the current group
        if arr[i] - current_group[0] <= interval:
            current_group.append(arr[i])
        else:
            groups.append(current_group)
            current_group = [arr[i]]

    # Add the last group
    groups.append(current_group)
    return groups

ediff = 2
batches = group_by_interval(energies, ediff)
print(f"Batches for ediff {ediff}")
for batch in batches:
    print(batch,len(batch),round(abs(batch[0]-batch[-1]),2))

# ....... for refinements make folders with roman numerals .....
romanList = ["i","ii","iii","iv","v","vi","vii","viii","ix","x",
        "xi","xii","xiii","xiv","xv","xvi","xvii","xviii","xix","xx",
        "xxi","xxii","xxiii","xxiv","xxv","xxvi","xxvii","xxviii","xxix","xxx",
        "xxxi","xxxii","xxxiii","xxxiv","xxxv","xxxvi"]

if len(romanList) != len(set(romanList)):
        raise AssertionError("Duplicate elements found in the romanlist.")

assert len(batches) <= len(romanList),"len(batches) - len(romanList)"
# ...................... existing roman serial folders ......
folderUpTo = ""
for item in romanList:
    if not os.path.exists(item):
        folderUpTo = item
        break
folders = romanList[romanList.index(folderUpTo):]
batches = batches[romanList.index(folderUpTo):]

# ...................... make driver files ......................
for num in range(len(batches)):    # each time make 2 folders
    path = f"{folders[num]}/"
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)
    fout = open("batchRoots.dat","w")
    batch = batches[num]
    for j in range(len(batch)):
        energy = batch[j]
        indx = energies.index(energy)
        fout.write(f"{energy} \t {files[indx]}\n")
    fout.close()
    os.chdir("../")

# .................. Clean folder .......................
fileList = ["jobname_SLURM.out","NODELIST"]
for item in fileList:
    if os.path.exists(item):
        os.remove(item)
# ................ EOF ..................................
