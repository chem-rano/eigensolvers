import numpy as np
import os
from ttns2.state import loadTTNSFromHdf5
import util
import warnings
from ttns2.driver import bracket
import sys

# ---------------- Function1: Reference TTNSs & energies -----------
def collect_ref(ref_dict={"ref_energy":None,"ref_coeffs":None}):
    ''' Collects reference energies and wavefunctions from the information 
    obtained from ref_dict
    Inputs:   ref_dict: Contains maximum 3 informations
              (a) path_to_ref : File location of the reference wavefunctions
              (b) ref_energy (optional): reference energy 
              (c) ref_coeffs optional): coeffecients for orthogonalied references
    Outputs: sorted energies, wavefunctions'''
    
    try:
        path = ref_dict["path_to_ref"]
    except KeyError:
        print("Key path_to_ref not found. Program terminated.")
        sys.exit(1)
    
    # collect wavefunction ane energies (if applicable)
    energies = []
    wavefunctions = []
    for iref in range(10000):
        try:
            filename = path + f"states/tns_{iref:05d}.h5"
            print(iref,filename)
            savedData = loadTTNSFromHdf5(filename)
            wavefunctions.append(savedData[0])
            energies.append(savedData[1]["energy"]) # in saved unit
        except FileNotFoundError:
            break
    assert iref < 10000,"Wavefunction collection loop does not break"
    
    # case of orthogonalized refernce wavefunctions
    if ref_dict["ref_coeffs"] is None:
        if ref_dict["ref_energy"] is None:
            indices = np.arange(0,len(energies),dtype=int)
            energies, indices = zip(*sorted(zip(energies, indices)))
            wavefunctions = wavefunctions[indices]
    
        elif ref_dict["ref_energy"] is not None: 
            energiesIn = ref_dict["ref_energy"]
            p = np.random.randint(0,len(energiesIn))
            checker = (loadTTNSFromHdf5(f"{path}/states/tns_{p:05d}.h5")[1]["energy"]== energies[p])
            checker = checker and (len(wavefunctions) == len(energiesIn))
            if checker:
                energies = energiesIn
            else:   # user given wrong energy file
                energies = energies
    
    # case of nonorthogonalized reference wavefunctions
    elif ref_dict["ref_coeffs"] is not None:
        if ref_dict["ref_energy"] is None:
            sys.exit("Orthogonalized reference energy file is not found") 

        elif ref_dict["ref_energy"] is not None: 
            energies = ref_dict["ref_energy"]
            assert len(wavefunctions) == len(energies)
        
            d = np.random.randint(0,len(energies))
            overlapd = 0.0
            num_ref = len(wavefunctions)
            ref_coeffs = ref_dict["ref_coeffs"]
            for i in range(num_ref):
                for j in range(num_ref):
                    overlapd += complex(ref_coeffs[i,d])*ref_coeffs[j,d]*bracket(wavefunctions[i], wavefunctions[j]) 
            assert overlapd == 1.0,"Self overlap of reference is not 1.0"
        
    return energies, wavefunctions
# ---------------- Function2: collect Krylov statetors ----------------
def assemble_krylov_statetors(cum_it,path_to_KS):
    ''' This function assembles all Krylov statetors at a particular cumulative
    iteration, cum_it and returns Krylov statetors as a list
    Inputs: cum_it -> cumulative iteration number
            path_to_KS -> file location to saved Krylov statetors
    Outputs: Ylist -> list of Krylov statetors'''

    Ylist = [] 
    for l in range(10000): # 10000 sufficiently large
        try:
            filename = path_to_KS + "tns_"+str(cum_it)+"_"+str(l)+".h5"
            Ylist.append(loadTTNSFromHdf5(filename)[0])
        except FileNotFoundError:
            break
    assert l < 10000,"Krylov space collection loop does not break"
    
    print("length of Ylist",len(Ylist))
    return Ylist

# ------- Function3: Lanczos eigenvalues and eigencofficients -------
def eigen_info(cum_it,path_to_KS):
    '''Collects eigen informations, eigenvalues and coefficients
    Inputs: cum_it -> cumulative iteration number
            path_to_KS -> file location to saved Krylov statetors
    Outputs: eigenvalues -> Lanczos eigenvalues
             coeff -> Lanczos eigen coefficients'''

    filename = path_to_KS + "tns_"+str(cum_it)+"_"+str(0)+".h5"
    coeff = loadTTNSFromHdf5(filename)[1]["eigencoefficients"]
    eigenvalues = loadTTNSFromHdf5(filename)[1]["eigenvalues"]
    return eigenvalues, coeff

# ------- Function4: Lanczos states for overlap calculation --------
def state_iterators(krylov_dim,states="all"):
    ''' Get indices for inner iteration loop
    (mainly to control statetor indices for overlap calculation)
    Helpful for terminated jobs due to computation time
    
    Inputs: krylov_dim -> Krylov dimension
            states (optional) -> str/ a list of index range
    Outputs: from_state -> beginning statetor index
             to_state -> end statetor index'''

    if states == "all":
        from_state = 0
        to_state = krylov_dim
    else:
        from_state = states[0] 
        to_state = states[1]
    
    if to_state == from_state:
        to_state = to_state + 1 # range to cover

    return from_state, to_state

def overlap_nonortho_ref(cum_it,Ylist,istate,refWF,path_nonortho_overlap):
    ''' Function to calculte/ upload overlap data of Lanczos states
    with  non-orthogonal refernce
    Inputs: cum_it -> cumulative iteration number
            Ylist -> List of Krylov vectors
            istate -> Lanczos state to calculate overlap
            refWF -> list of nonorthogonal references
            path_nonortho_overlap -> data file containing overlap 
            data (if available)
    Outputs: overlap -> overlap values as 1D numpy array
             overlap2 -> squared overalp values as 1D numpy array
             total -> total squared overlap'''

    num_ref = len(refWF)
    mstates = len(Ylist)
    if path_nonortho_overlap is None:
        total = 0.0
        overlap = np.empty(num_ref,dtype=float)
        overlap2 = np.empty(num_ref,dtype=float)
        for num in range(num_ref):
            for k in range(mstates):
                overlap[num] += bracket(refWF[num], Ylist[k])*coeffs[k,state]
            overlap2[num] = overlap[num]*overlap[num]
            total += overlap2[num]

    elif path_nonortho_overlap is not None:
        filename = f"{path_nonortho_overlap}/Overlap_it{cum_it}_vec{istate}.dat"
        overlap = np.loadtxt(filename,usecols=(3),skiprows=1,dtype=float)
        overlap2 = np.loadtxt(filename,usecols=(4),skiprows=1,dtype=float)
        total = np.loadtxt(filename,usecols=(5),skiprows=1,dtype=float)[-1]

    return overlap, overlap2, total

# --------- overlap, squared overlap and sum of squared overlap -----
def calculate_and_write_overlap(start_cum,max_cum,path_to_KS,states_selected,ref_dict,
        path_nonortho_overlap=None):
    ''' Main function to calculate and write overlap, squared overlap,
    and sum of squared overlap
    
    -------------  Equations ------------------
    ovlp_pq = <Ref_p|psi_q> = <Ref_p|\sum_k(cp_kq*phi_k)>
    If more accurate references are available through orthogonalization of {|Ref>}
    That is |RefO_p> = \sum_l(cr_lp*|Ref_l>)
    Then ovlpO_pq = <RefO_p|psi_q> = <RefO_p|\sum_k(cp_kq*phi_k)>
                                   = \sum_l(cr*_lp)<Ref_l|\sum_k(cp_kq*phi_k)> ... (1)

    Special case of having overlap data for nonorthogonal refererence
    That means, ovlp is available then relation to get ovlpO is:
    ovlpO_pq = <RefO_p|psi_q> 
             = <\sum_l(cr_lp)Ref_l|psi_q>
             = \sum_l(cr*_lp)ovlp_lq .............. (2)
    -------------------------------------------
    Inputs: start_cum -> starting cumulative iteration 
            max_cum -> maximum cumulative iteration
            path_to_KS -> path to saved Krylov statetors
            states_selected -> Lanczos states; 'all' as string/a list of indices [0,2]
            path_nonortho_overlap -> nonorthogonal overlap file, if provided then overlap is 
                                calculated using equation 2
    
    Generated file -> Overlap_it{i}_state{state}.out for iteration "i" and state "state"
    '''
    
    refE, refWF = collect_ref(ref_dict) 
    num_ref = len(refE)
    num_ref = 10 #NOTE
    ref_in_cm = util.au2unit(np.array(refE),"cm-1")

    for it in range(start_cum,max_cum+1):
        Ylist = assemble_krylov_statetors(it,path_to_KS)
        eigenvalues, coeffs = eigen_info(it,path_to_KS)
        ev_in_cm = util.au2unit(np.array(eigenvalues),"cm-1")
        mstates = coeffs.shape[1]
        assert(mstates == len(Ylist))
   
        from_state, to_state = state_iterators(mstates,states_selected)
        for istate in range(from_state,to_state):
    
            overlap_file = open(f"Overlap_it{i}_vec{state}.out","w")
            output_format = "{:>6} {:>16} {:>16} {:>16} {:>16} {:>16}" 
            lines = output_format.format("Index","RefE","eigenvalue","overlap","overlap-squared","Total\n")
            overlap_file.write(lines)

            overlap, overlap2, total = overlap_nonortho_ref(it,Ylist,istate,refWF,path_nonortho_overlap)
            
            if ref_dict["ref_coeffs"] is not None:
                totalC = 0.0
                overlapC = np.empty(num_ref,dtype=float)
                overlap2C = np.empty(num_ref,dtype=float)
                ref_coeffs = ref_dict["ref_coeffs"]
                is_real = np.isreal(ref_coeffs).all()
                for num in range(num_ref):
                    for ic in range(num_ref):
                        if is_real:
                            overlapC[num] += ref_coeffs[ic,num] * overlap[num] #NOTE coeffs
                        else:
                            overlapC[num] += complex(ref_coeffs[ic,num]) * overlap[num]
                overlap2C[num] = overlapC[num]*overlapC[num]
                total += overlap2C[num]

        
            lines = "{:>6}".format(num)
            lines += "{:>16}".format(f"{ref_in_cm[num,0]:.6f}") #NOTE
            lines += "{:>16}".format(f"{ev_in_cm[istate]:.6f}")
            lines += "{:>16}".format(f"{overlap:.4f}")
            lines += "{:>16}".format(f"{overlap2:.4f}")
            lines += "{:>16}".format(f"{total:.4f}")+"\n"
            overlap_file.write(lines)
            overlap_file.close()

            assert total <= 1.0, f"Total squared overlap {total} greater than 1.0"
            assert totalC <= 1.0, f"Total squared overlap {total} greater than 1.0"

# --------------------------- Main function ------------------------
if __name__ == "__main__":
    start_cum, max_cum = 1, 3 # cumulative iterations
    path_to_KS = "demo/saveTNSs/" # path to saved Krylov statetors
    states = 'all'     # Lanczos states; 'all' as string/a list of indices [0,2]
    
    path_to_ref = "/data/larsson/Eigen/RUNS/tns_D70/" # Path to References
    ref_energy = np.load(f"{path_to_ref}/matrices/evuv.npz")["ev"] # reference energies
    ref_coeffs = np.load(f"{path_to_ref}/matrices/evuv.npz")["uv"]
    ref_dict = {"path_to_ref":path_to_ref,"ref_energy":ref_energy,"ref_coeffs":ref_coeffs}
   
    calculate_and_write_overlap(start_cum,max_cum,path_to_KS,states,ref_dict)
# ---------------  EOF ----------------------------------------------
