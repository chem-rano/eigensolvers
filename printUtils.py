import util
from datetime import datetime
from util_funcs import find_nearest

# -----------------------------------------------------
def convert(arr,eShift,convertUnit):
    ''' For converting ndarray (energy or matrix) with 
    adjusted eShift and unit conversion'''
    arrShifted = None 
    if convertUnit == 'au':
        arrShifted = arr-eShift
    else:
        arrShifted = util.au2unit(arr,convertUnit)-eShift
    return arrShifted

# -----------------------------------------------------

def fileHeader(fstring,sigma,zpve,L,maxit,D,eConv,options,guess="Random",printInfo=False):
    """ Prints header with all input informations 
    printInfo prints this header to the screen, recorded in sweepOutputs"""
    
    fout = open("iterations.out","a");fplot = open("data2Plot.out","a")
    file = fout if fstring == "out" else fplot

    optionsOrtho = options["orthogonalizationArgs"]
    optionsLinear= options["linearSystemArgs"]
    optionsFitting = options["stateFittingArgs"]
    
    solver = optionsLinear["iterativeLinearSystemOptions"].solver
    siteLinearTol = optionsLinear["iterativeLinearSystemOptions"].tol
    globalLinearTol = optionsLinear["convTol"]
    nsweepLinear = optionsLinear["nSweep"]
    
    fittingTol = optionsFitting["convTol"]
    nsweepFitting = optionsFitting["nSweep"]
    
    if fstring == "plot":    # for data extractor code
        line = "startingPoint"
        file.write(line+"\n")

    dateTime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    dt_string = "\t\t"+dateTime+"\t\t\n"
    line = "*"*70 + "\n\t\tStarting computation\t\t\n"+dt_string+"*"*70+"\n"
    file.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:6} {:10} :: {:20}".format("sigma",sigma,"Target")
    file.write(line+"\n")
    if printInfo:
        print(line)
    
    line = "{:6} {:10} :: {:20}".format("L",L,"Krylov space")
    file.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:6} {:10} :: {:20}".format("maxit",maxit,"Maximum Lanczos iterations")
    file.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:6} {:10} :: {:20}".format("econv",eConv,"Eigenvalue convergence")
    file.write(line+"\n")
    if printInfo:
        print(line)
    
    line = "{:6} {:10} :: {:20}".format("D",D,"Bond dimension")
    file.write(line+"\n")
    if printInfo:
        print(line)
    
    line = "{:6} {:10} :: {:20}".format("ftol",fittingTol,"Fitting Tolerance")
    file.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:6} {:10} :: {:20}".format("fsweep",nsweepFitting,"Number of sweeps: fitting")
    file.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:6} {:10} :: {:20}".format("ltol1",siteLinearTol,"Site tolerance: Linear solver")
    file.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:6} {:10} :: {:20}".format("ltol2",globalLinearTol,"global tolerance: Linear solver")
    file.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:6} {:10} :: {:20}".format("lsweep",nsweepLinear,"Number of sweeps: Linear solver")
    file.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:6} {:10} :: {:20}".format("solver",solver,"Linear solver")
    file.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:6} {:10} :: {:20}".format("D",guess,"Guess TTNS\n\n")
    file.write(line+"\n")
    if printInfo:
        print(line)
    
    if fstring == "plot":
        line = "it\ti\tnCum\tev_nearest\tcheck_ev\trel_ev\ttime\n"
        file.write(line)
    
    fout.close()
    fplot.close()

def fileFooter(fstring,printInfo=False):
    """ Prints footer with job complete message
    printInfo prints this footer to the screen, recorded in sweepOutputs"""
    
    fout = open("iterations.out","a");fplot = open("data2Plot.out","a")
    file = fout if fstring == "out" else fplot
    
    if fstring == "plot":    # for data extractor code
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

    fout = open("iterations.out","a");fplot = open("data2Plot.out","a")
    file = fout if fstring == "out" else fplot
    
    if choices=None:
        eShift = 0.0; convertUnit="au"
    else:
        eShift = choices["eShift"];convertUnit = choices["convertUnit"]
    
    if len(args) == 2:
        if args[0] == "OVERLAP MATRIX":
            file.write("OVERLAP MATRIX\n")
            file.write(f"{args[1]}")
        
        elif args[0] == "HAMILTONIAN MATRIX":
            file.write("HAMILTONIAN MATRIX\n")
            file.write(f"{convert(args[1],eShift,convertUnit)}")
        
        elif args[0] == "Eigenvalues":
            file.write("Eigenvalues\n")
            file.write(f"{convert(args[1],eShift,convertUnit)}")
        

    
    elif len(args)> 2 and args[0] == "iteration details":
        line =  "\n\n...................\tInfo per iteration\t...................\n"
        line += "Lanczos iteration: "+str(args[1]+1)+"\tKrylov iteration: "+str(args[2])
        line += "\tCumulative Krylov iteration: "+str(args[3])
        file.write(line+"\n")

    elif len(args)> 2 and args[0] == "Final results":
        line = "\n\n"+"-"*20+"\tFINAL RESULTS\t"+"-"*20+"\n"
        file.write(line)
        energies = convert(args[1],eShift,convertUnit)
        target = args[2]
        ev_nearest = find_nearest(energies,target)[1]
        file.write("{:30} :: {: <4}, {: <4}".format("Target, calculated nearest",target,round(ev_nearest),4)+"\n")
        list_results = ""
        for i in range(0,(len(energies)-1),1):
            list_results += str(round(energies[i],4))+", "
        list_results +=  str(round(energies[-1],4))
        file.write("{:30} :: {: <4}".format("All subspace eigenvalues",list_results)+"\n")

    else:           
        for item in args:
            file.write(str(item)+" ")
    file.write("\n")
    fout.close()
    fplot.close()
