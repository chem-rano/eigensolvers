import numpy as np
import os
import util
import warnings

# -------------------------------------------------------------------
# This script does sorting of overlap data 
# and is written as per MR's folder arrangement

# ---------------- Load data to be sorted from overlap file ---------
imin, imax = 1, 1000
jmin, jmax = 0, 1000
dirSplits = os.getcwd().split("/")
dirSplits = ["","overlapCorrected"] #NOTE for demo

if dirSplits[-1] == "overlapCorrected":
    both = False
    jmin, jmax = 0,1
    s1, s2 = "",""
    D = 70
if dirSplits[-2] == "overlapCorrected" and dirSplits[-1] =="ref150":
    both = False
    jmin, jmax = 0,1
    s1,s2 = "../","ref150/"
    D = 150
elif dirSplits[-1] == "allOthersCorrected":
    both = True
    s1, s2 = "",""
    D = 70
elif dirSplits[-2] == "allOthersCorrected" and dirSplits[-1] =="ref150":
    both = True
    s1,s2 = "../","ref150/"
    D = 150

both = True; jmin, jmax = 0, 1000; D = 70 # NOTE for demo only
# ---------------- Reference TTNSs order-----------------------------
refOrderFile = f"/home/madhumitarano/data/PR39/Eigen/forPaper/EigenRefOrder/AccordanceREF{D}.dat"
correctOrder = list(np.genfromtxt(refOrderFile,skip_header=1,skip_footer=2,usecols=(2),dtype=int))
numRef = len(correctOrder)
# -------------------------------------------------------------------
breakInnerLoop = False
for i in range(imin,imax):
    for j in range(jmin,jmax):
        try:
            if not both: 
                oldFile = f"{s1}../overlap/{s2}Overlap{i}.out"
                newFile = f"Overlap_it{i}_vec0.out"
            elif both:
                #oldFile = f"{s1}../allOthers/{s2}Overlap_it{i}_vec{j}.out"
                #newFile = f"Overlap_it{i}_vec{j}.out"
                oldFile = f"check_overlap/unsorted/Overlap_it{i}_vec{j}.out" #NOTE for demo only
                newFile = f"check_overlap/sorted/Overlap_it{i}_vec{j}.out" #NOTE for demo only
            print(oldFile,newFile)

            files = open(oldFile);files.close() # loadtxt not raising fileNotFound Error
            overlapData = np.loadtxt(oldFile,usecols=(0,1,2,3,4,5),skiprows=1)
            numRef = len(overlapData[:,0]) # some are truncated
            correctOrder = list(np.genfromtxt(refOrderFile,skip_header=1,skip_footer=2,usecols=(2),dtype=int))
            correctOrder = correctOrder[0:numRef]
           
            overlapData[:,3] = overlapData[:,3][correctOrder]
            overlapData[:,4] = overlapData[:,4][correctOrder]

            overlap_file = open(newFile,"w")
            lines = "{:>6} {:>16} {:>16} {:>16} {:>16} {:>16}".format("Index","RefE",\
            "eigenvalue","overlap","overlap-squared","Total")
            lines += "\n"
            overlap_file.write(lines) 
            
            total = 0.0
            for num in range(numRef):
                total += overlapData[num,4]

                lines = "{:>6}".format(num)
                lines += "{:>16}".format(f"{overlapData[num,1]:.6f}")
                lines += "{:>16}".format(f"{overlapData[num,2]:.6f}")
                lines += "{:>16}".format(f"{overlapData[num,3]:.4f}")
                lines += "{:>16}".format(f"{overlapData[num,4]:.4f}")
                lines += "{:>16}".format(f"{total:.4f}")+"\n"
                overlap_file.write(lines)

            overlap_file.close()
        
        except FileNotFoundError:
            breakInnerLoop= True
            break
        
    if j<2 and breakInnerLoop: 
        break
# -------------------------------------------------------
