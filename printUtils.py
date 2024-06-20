import util
from datetime import datetime
from util_funcs import find_nearest
import numpy as np
from ttns2.diagonalization import IterativeLinearSystemOptions

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

def fileHeader(fstring,options,sigma,zpve,L,maxit,eConv,
        D=None,guess="Random",printInfo=True):
    """ Prints header with all input informations 
    printInfo prints this header to the screen, recorded in sweepOutputs"""
    
    fout = open("iterations.out","a");fplot = open("data2Plot.out","a")
    file = fout if fstring == "out" else fplot

    optionsOrtho = options["orthogonalizationArgs"]
    optionsLinear = options["linearSystemArgs"]
    optionsFitting = options["stateFittingArgs"]
   
    siteOptions = optionsLinear["iterativeLinearSystemOptions"]
    if isinstance(siteOptions, IterativeLinearSystemOptions):
        solver = optionsLinear["iterativeLinearSystemOptions"].solver
        siteLinearTol = optionsLinear["iterativeLinearSystemOptions"].tol
    else:
        solver = optionsLinear["linearSolver"]
        siteLinearTol = optionsLinear["linear_tol"]

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
    
    if D is not None:
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

    line = "{:6} {:>10} :: {:20}".format("Guess",guess,"Guess TTNS\n\n")
    file.write(line+"\n")
    if printInfo:
        print(line)
    
    if file == fplot:
        line = "it\ti\tnCum\tev_nearest\t\tabs_ev\t\trel_ev\t\ttime (seconds)\n"
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

def _outputFile(status,args):
    file = open("iterations.out","a")
    
    if args[0] == "overlap":
        file.write("OVERLAP MATRIX\n")
        file.write(f"{args[1]}")
        file.write("\n")
        file.write("\n")
        
    elif args[0] == "hamiltonian":
        file.write("HAMILTONIAN MATRIX\n")
        hmat = convert(args[1],status["eShift"],status["convertUnit"])
        file.write(f"{hmat}")
        file.write("\n")
        file.write("\n")
        
    elif args[0] == "eigenvalues":
        file.write("Eigenvalues\n")
        evalues = convert(args[1],status["eShift"],status["convertUnit"])
        file.write(f"{evalues}")
        file.write("\n")
    
    elif args[0] == "results":
        # same as 'eigenvalues', with ev_nearest and final message
        file.write("\n")
        file.write("\n")
        file.write("-"*20)
        file.write("\tFINAL RESULTS\t")
        file.write("-"*20)
        file.write("\n")
        energies = convert(args[1],status["eShift"],status["convertUnit"])
        file.write("All subspace eigenvalues:")
        file.write("\n")
        file.write(f"{energies}")
        file.write("\n")
        assert len(args) > 2; target = args[2]
        ev_nearest = find_nearest(energies,target)[1]
        file.write("Target, Lanczos (nearest)")
        file.write("{target}, {ev_nearest}")
        file.write("\n")
        
    elif args[0] == "iteration":
        file.write("\n")
        file.write("\n")
        file.write("."*20)
        file.write("\tInfo per iteration\t")
        file.write("."*20)
        file.write("\n")
        file.write("Lanczos iteration: "+str(status["outerIter"]+1))
        file.write("\tKrylov iteration: "+str(status["microIter"]))
        file.write("\tCumulative Krylov iteration: "+str(status["cumIter"]))
        file.write("\n")
    file.close()


def _plotFile(status,args):

    file = open("data2Plot.out","a")
    
    it = status["outerIter"]
    i = status["microIter"]
    nCum = status["cumIter"]
    runTime = status["runTime"]
    evalue = util.au2unit(args[0],status["convertUnit"])
    ref = util.au2unit(args[1],status["convertUnit"])
    abs_diff = np.abs(evalue - ref)
    rel_ev = abs_diff/np.abs(evalue)
    exEnergy = evalue - status["eShift"]
    file.write(f'{it}\t{i}\t{nCum}\t')
    file.write(f'{exEnergy}\t{abs_diff}\t{rel_ev}\t{runTime}\n')
    file.close()

def writeFile(filestring,status,*args):
    """ A single print function for overlap, Hamitonian matrix, iteration details 
    and final eigenvalue"""
    
    if filestring == "out":
        _outputFile(status,args)
    elif filestring == "plot":
        _plotFile(status,args)
