import util
from datetime import datetime
from util_funcs import find_nearest
import numpy as np

# -----------------------------------------------------
def convert(arr,eShift=0.0,unit='au'):
    ''' For converting ndarray (energy or matrix) with 
    adjusted eShift and unit conversion'''

    arrShifted = None
    if unit == 'au':
        arrShifted = arr - eShift
    else:
        arrShifted = util.au2unit(arr,unit)-eShift
    return arrShifted

# -----------------------------------------------------

def fileHeader(fstring,sigma,zpve,L,maxit,D,eConv,options,guess="Random",printInfo=True):
    """ Prints header with all input informations 
    printInfo prints this header to the screen, recorded in sweepOutputs"""
    
    fout = open("iterations.out","a");fplot = open("data2Plot.out","a")
    file = fout if fstring == "out" else fplot

    optionsOrtho = options["orthogonalizationArgs"]
    optionsLinear = options["linearSystemArgs"]
    optionsFitting = options["stateFittingArgs"]
    
    solver = optionsLinear["iterativeLinearSystemOptions"].solver
    siteLinearTol = optionsLinear["iterativeLinearSystemOptions"].tol
    globalLinearTol = optionsLinear["convTol"]
    nsweepLinear = optionsLinear["nSweep"]
    
    fittingTol = optionsFitting["convTol"]
    nsweepFitting = optionsFitting["nSweep"]
    
    if file == fplot:    # for data extractor code
        line = "startingPoint"
        file.write(line+"\n")

    dateTime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    dt_string = "\t\t"+dateTime+"\t\t\n"
    line = "*"*70 + "\n\t\tStarting computation\t\t\n"+dt_string+"*"*70+"\n"
    file.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:6} {:>10} :: {:20}".format("sigma",sigma,"Target")
    file.write(line+"\n")
    if printInfo:
        print(line)
    
    line = "{:6} {:>10} :: {:20}".format("L",L,"Krylov space")
    file.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:6} {:>10} :: {:20}".format("maxit",maxit,"Maximum Lanczos iterations")
    file.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:6} {:>10} :: {:20}".format("econv",eConv,"Eigenvalue convergence")
    file.write(line+"\n")
    if printInfo:
        print(line)
    
    line = "{:6} {:>10} :: {:20}".format("D",D,"Bond dimension")
    file.write(line+"\n")
    if printInfo:
        print(line)
    
    line = "{:6} {:>10} :: {:20}".format("ftol",fittingTol,"Fitting Tolerance")
    file.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:6} {:>10} :: {:20}".format("fsweep",nsweepFitting,"Number of sweeps: fitting")
    file.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:6} {:>10} :: {:20}".format("ltol1",siteLinearTol,"Site tolerance: Linear solver")
    file.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:6} {:>10} :: {:20}".format("ltol2",globalLinearTol,"global tolerance: Linear solver")
    file.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:6} {:>10} :: {:20}".format("lsweep",nsweepLinear,"Number of sweeps: Linear solver")
    file.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:6} {:>10} :: {:20}".format("solver",solver,"Linear solver")
    file.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:6} {:>10} :: {:20}".format("D",guess,"Guess TTNS\n\n")
    file.write(line+"\n")
    if printInfo:
        print(line)
    
    if file == fplot:
        line = "it\ti\tnCum\tev_nearest\tcheck_ev\trel_ev\ttime\n"
        file.write(line)
    
    fout.close()
    fplot.close()

def fileFooter(fstring,printInfo=True):
    """ Prints footer with job complete message
    printInfo prints this footer to the screen, recorded in sweepOutputs"""
    
    fout = open("iterations.out","a");fplot = open("data2Plot.out","a")
    file = fout if fstring == "out" else fplot
    
    if file == fplot:    # for data extractor code
        line = "endingPoint"
        file.write(line+"\n")
    
    dateTime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    dt_string = "\t\t"+dateTime+"\t\t\n"
    line = "\n"+"*"*70 + "\n\t\tEnd of computation\t\t\n"+dt_string+"*"*70+"\n\n"
    file.write(line+"\n")
    if printInfo:
        print(line)
    
    fout.close()
    fplot.close()


def writeFile(fstring,*args,choices=None):
    """ A single print function for overlap, Hamitonian matrix, iteration details 
    and final eigenvalue
    (May be split into different functions for better readability)"""

    file = open("iterations.out","a")
    
    if choices == None:
       choices = { "eShift":0.0, "convertUnit":"au"} 
    
    if args[0] == "overlap":
        file.write("OVERLAP MATRIX\n")
        file.write(f"{args[1]}")
        
    elif args[0] == "hamiltonian":
        file.write("HAMILTONIAN MATRIX\n")
        hmat = convert(args[1],choices["eShift"],choices["convertUnit"])
        file.write(f"{hmat}")
        
    elif args[0] == "eigenvalues":
        file.write("Eigenvalues\n")
        evalues = convert(args[1],choices["eShift"],choices["convertUnit"])
        file.write(f"{evalues}")
        

    elif args[0] == "results":
        # same as 'eigenvalues', with ev_nearest and final message
        file.write("\n\n"+"-"*20+"\tFINAL RESULTS\t"+"-"*20+"\n")
        energies = convert(args[1],choices["eShift"],choices["convertUnit"])
        assert len(args) > 2; target = args[2]
        ev_nearest = find_nearest(energies,target)[1]
        file.write("{:20} :: {:<.4f}, {:<.4f}".format("Target, Lanczos (nearest)",
            target,round(ev_nearest),4)+"\n")
        file.write("{:20} :: {}".format("All subspace eigenvalues",energies)+"\n")
    
    elif args[0] == "iteration":
        # I like the first formatted string in the output :)
        assert len(args) > 2
        line =  "\n\n...................\tInfo per iteration\t...................\n"
        line += "Lanczos iteration: "+str(args[1]+1)+"\tKrylov iteration: "+str(args[2])
        line += "\tCumulative Krylov iteration: "+str(args[3])
        file.write(line+"\n")
    
    else:           
        for item in args:
            file.write(str(item)+" ")
    file.write("\n")
    file.close()

def writePlotfile(status,evalue,ref):
    file = open("data2Plot.out","a")
    
    it = status["iteration"]
    i = status["microIteration"]
    nCum = status["cummulativeIteration"]
    runTime = status["endTime"]-status["startTime"]
    abs_diff = np.abs(evalue - ref)
    rel_ev = abs_diff/np.abs(evalue)
    file.write(f'{it}\t{i}\t{nCum}\t')
    file.write(f'{evalue}\t{abs_diff}\t{rel_ev}\t{runTime}\n')
    file.close()
