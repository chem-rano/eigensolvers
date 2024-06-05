import util

#filepath = "/home/madhumita/mr/zundel/gitspace/eigensolvers/examples/"
#inputs  = open(filepath+"printChoicesNumpyvector").readlines()
#convertUnit = (inputs[0].split("#"))[0];#print(bool(convertUnit))
#eShift = float((inputs[1].split("#"))[0]);#print(eShift)

# -----------------------------------------------------
def convert(arr,eShift =0.0, convertUnit="au"):
    ''' For converting ndarray (energy or matrix) with 
    adjusted eShift and unit conversion'''
    arrShifted = None 
    if convertUnit == 'au':
        arrShifted = arr-eShift
    else:
        arrShifted = util.au2unit(arr,convertUnit)-eShift
    return arrShifted

# -----------------------------------------------------

def writeInputs(fout,dateTime,sigma,zpve,L,maxit,D,eConv,options,guess="Random",printInfo=False):
    optionsOrtho = options["orthogonalizationArgs"]
    optionsLinear= options["linearSystemArgs"]
    optionsFitting = options["stateFittingArgs"]
    
    siteLinearTol = optionsLinear["iterativeLinearSystemOptions"][0]
    globalLinearTol = optionsLinear["convTol"]
    nsweepLinear = optionsLinear["nSweep"]
    
    fittingTol = optionsFitting["convTol"]
    nsweepFitting = optionsFitting["nSweep"]


    dt_string = "\t\t"+dateTime+"\t\t\n"
    line = "*"*70 + "\n\t\tStarting computation\t\t\n"+dt_string+"*"*70+"\n"
    fout.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:60} :: {: <4}, {: <4}".format("Krylov space and Lanczos iterations",L,maxit)
    fout.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:60} :: {: <4}, {: <4}, {: <4}".format("Linear solver (siteLinearTol, globalLinearTol,nsweepLinear)",siteLinearTol,globalLinearTol,nsweepLinear)
    fout.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:60} :: {: <4}, {: <4}".format("Fitting solver (fittingTol,nsweepFitting)",fittingTol,nsweepFitting)
    fout.write(line+"\n")
    if printInfo:
        print(line)

    line = "{:60} :: {: <4}".format("Target (sigma)",sigma)
    fout.write(line+"\n")
    if printInfo:
        print(line)
    
    line = "{:60} :: {: <4}".format("eConv (cm-1)",eConv)
    fout.write(line+"\n")
    if printInfo:
        print(line)
    
    line = "{:60} :: {: <4}".format("Bond dimension (D)",D)
    fout.write(line+"\n")
    if printInfo:
        print(line)
    
    line = "{:60} :: {: <4}".format("Guess TTNS",guess)+"\n\n"
    fout.write(line+"\n")
    if printInfo:
        print(line)

def printfooter(fout,printInfo=False,dateTime=None):
    line = "\n"+"*"*70 + "\n\t\tEnd of computation\t\t\n"+"*"*70+"\n\n"
    fout.write(line+"\n")
    if printInfo:
        print(line)

def fplotHeader(fout,dateTime,sigma,zpve,L,maxit,D,eConv,options):
    optionsOrtho = options["orthogonalizationArgs"]
    optionsLinear= options["linearSystemArgs"]
    optionsFitting = options["stateFittingArgs"]
    
    siteLinearTol = optionsLinear["iterativeLinearSystemOptions"][0]
    globalLinearTol = optionsLinear["convTol"]
    nsweepLinear = optionsLinear["nSweep"]
    
    fittingTol = optionsFitting["convTol"]
    nsweepFitting = optionsFitting["nSweep"]

    line = "startingPoint"
    fout.write(line+"\n")
    dt_string = "\t\t"+dateTime+"\t\t\n"
    line = "*"*70 + "\n\t\tStarting computation\t\t\n"+dt_string+"*"*70+"\n"
    fout.write(line+"\n")

    line = "{:2}, {:10} #{:10}".format("L",L,"Total Krylov dim")
    fout.write(line+"\n")
    line = "{:2}, {:10} #{:10}".format("I",maxit,"maxit")
    fout.write(line+"\n")

    line = "{:2}, {:10} #{:10}".format("t",siteLinearTol,"siteLinearTol")
    fout.write(line+"\n")
    line = "{:2}, {:10} #{:10}".format("T",globalLinearTol,"globalLinearTol")
    fout.write(line+"\n")
    line = "{:2}, {:10} #{:10}".format("n",nsweepLinear,"nsweepLinear")
    fout.write(line+"\n")

    line = "{:2}, {:10} #{:10}".format("f",fittingTol,"fittingTol")
    fout.write(line+"\n")
    line = "{:2}, {:10} #{:10}".format("F",nsweepFitting,"nsweepFitting")
    fout.write(line+"\n")

    line = "{:2}, {:10} #{:10}".format("S",sigma,"Sigma")
    fout.write(line+"\n")

    line = "{:2}, {:10} #{:10}".format("e",eConv,"eConv in cm-1")
    fout.write(line+"\n")

    line = "{:2}, {:10} #{:10}".format("D",D,"Bond dimension")
    fout.write(line+"\n\n")
    line = "it\ti\tnCum\tev_nearest\tcheck_ev\trel_ev\ttime\n"
    fout.write(line)

def fplotFooter(fout):
    line = "endingPoint"
    fout.write(line+"\n")
    line = "\n"+"*"*70 + "\n\t\tEnd of computation\t\t\n"+"*"*70+"\n\n"
    fout.write(line+"\n")

def writeFile(fstring,*args,sep=" ",endline=True):
    fout = open("iterations.out","a");fplot = open("data2Plot.out","a")
    file = fout if fstring == "out" else fplot
    
    if len(args) == 2:
        if args[0] == "OVERLAP MATRIX":
            file.write("OVERLAP MATRIX\n")
            file.write(f"{args[1]}")
        elif args[0] == "HAMILTONIAN MATRIX":
            file.write("HAMILTONIAN MATRIX\n")
            file.write(f"{convert(args[1])}")
        elif args[0] == "Eigenvalues":
            file.write("Eigenvalues\n")
            file.write(f"{convert(args[1])}")
    
    elif len(args)> 2 and args[0] == "iteration details":
        line = "Lanczos iteration\t"+str(args[1]+1)+"\tKrylov iteration\t"+str(args[2])
        line += "\tCumulative Krylov iteration\t"+str(args[3])
        file.write(line+"\n")
    else:                           # whatever else write with \t separation
        for item in args:
            file.write(str(item)+sep)
    file.write("\n") if endline else file.write("\t")
    fout.close()
    fplot.close()
