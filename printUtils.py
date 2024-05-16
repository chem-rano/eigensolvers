import util

def writeInfo(fout,dateTime,sigma,zpve,L,maxit,D,eConv,options,guess="Random",printInfo=False):
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

def fplotFooter(fout):
    line = "endingPoint"
    fout.write(line+"\n")
    line = "\n"+"*"*70 + "\n\t\tEnd of computation\t\t\n"+"*"*70+"\n\n"
    fout.write(line+"\n")

def _writeFile(file,*args,sep=" ",endline=True):
    if len(args) ==1:
        file.write(f"{args[0]}")
    elif len(args)> 1:
        for item in args:
            file.write(str(item)+sep)
    file.write("\n") if endline else file.write("\t")
