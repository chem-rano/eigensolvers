import numpy as np
import os
from ttns2.state import loadTTNSFromHdf5
import util
import warnings
from ttns2.driver import bracket
import sys
import time

# ---------------- Function1: Reference TTNSs & energies -----------
def collect_ref(ref_dict):
    ''' Collects reference energies and wavefunctions from the information 
    given in the "ref_dict" dictionary
    Additionally, this function scrutinizes reference information to be correct 
    to some extent
    Inputs:   ref_dict: Dictionary containing REF info (contains maximum 5 keys)
              (a) path_to_ref : file location of the reference wavefunctions
              (b) ref_energy (optional): reference energy 
              (c) energy_unit (optional): reference energy unit
              (d) zpve (optional): zero-point energy
              (e) ref_coeffs (optional): coeffecients for orthogonalized references
    Note: ref_energy & zpve must be in same unit i.e., in energy_unit
    Outputs: sorted reference energies, reference wavefunctions'''
    
    # Without specification of reference wavefunctions
    try:
        path = ref_dict["path_to_ref"]
    except KeyError:
        print("Key path_to_ref not found. Program terminated.")
        sys.exit(1)
    
    #ref_dict with defaults 
    default_dict = {"ref_energy":None,"ref_coeffs":None, "energy_unit":"au","zpve":0.0}
    for key in default_dict:
        if key not in ref_dict: 
            ref_dict[key] = default_dict[key]
    
    # In case of orthogonalized reference, without specification of energies
    if ref_dict["ref_coeffs"] is not None and ref_dict["ref_energy"] is None:
        sys.exit("Orthogonalized reference energy file is not found")
    
    # collect wavefunction and energies
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
    
    # case1: nonorthogonalized reference wavefunctions
    if ref_dict["ref_coeffs"] is None:
        if ref_dict["ref_energy"] is None:
            energies, wavefunctions = zip(*sorted(zip(energies, wavefunctions)))
    
        elif ref_dict["ref_energy"] is not None: 
            energiesIn = np.array(ref_dict["ref_energy"]) + ref_dict["zpve"]
            if ref_dict["energy_unit"] != "au":
                energiesIn = util.unit2au(np.array(ref_dict["ref_energy"]),ref_dict["energy_unit"])
            indices = np.arange(0,len(energies),dtype=int)
            energies, indices = zip(*sorted(zip(energies, indices)))
            equal = (list(energiesIn) == energies)

            if not equal: # user given wrong/ unsorted energy file
                energies, wavefunctions = zip(*sorted(zip(energies, wavefunctions)))
    
    # case2: orthogonalized refernce wavefunctions
    elif ref_dict["ref_coeffs"] is not None:
        energies = ref_dict["ref_energy"]
        assert len(wavefunctions) == len(energies)
       
        if False: # test when going to a new case
            # (a) self-overlap
            d = np.random.randint(0,len(energies))
            ref_coeffs = ref_dict["ref_coeffs"]
            num_ref = len(wavefunctions)
            overlapd = np.array(0, dtype=wavefunctions[0].rootNode.dtype)
            is_real = np.isreal(ref_coeffs).all()
            t1 = time.time()
            for i in range(num_ref):
                for j in range(num_ref):
                    if is_real:
                        overlapd += ref_coeffs[i,d]*ref_coeffs[j,d]*bracket(wavefunctions[i], wavefunctions[j])
                    else:
                        overlapd += complex(ref_coeffs[i,d])*ref_coeffs[j,d]*bracket(wavefunctions[i], wavefunctions[j]) 
                    print(i,j,overlapd)
            print(f"Self overlap of reference,{d} is {overlapd} (overlapd should be 1.0)")
            print("Self overlap computation time",time.time() -t1) #Example:1.0000081082944656 in 133349 sec
            # (b) energy check: #NOTE 
    return energies, wavefunctions
# ---------------- Function2: collect Krylov statetors ----------------
def assemble_krylov_vectors(cum_it,path_to_KS):
    ''' This function assembles all Krylov vectors at a particular cumulative
    iteration, cum_it and returns Krylov vectors as a list
    Inputs: cum_it -> cumulative iteration number
            path_to_KS -> file location to saved Krylov statetors
    Outputs: Ylist -> list of Krylov vectors'''

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
            path_to_KS -> file location to saved Krylov vectors 
    Outputs: eigenvalues -> Lanczos eigenvalues
             coeff -> Lanczos eigen coefficients'''

    filename = path_to_KS + "tns_"+str(cum_it)+"_"+str(0)+".h5"
    coeff = loadTTNSFromHdf5(filename)[1]["eigencoefficients"]
    eigenvalues = loadTTNSFromHdf5(filename)[1]["eigenvalues"]
    return eigenvalues, coeff

# ------- Function4: Lanczos states for overlap calculation --------
def state_iterators(krylov_dim,states="all"):
    ''' Get indices for inner iteration loop
    (mainly to control state indices for overlap calculation)
    Helpful for terminated jobs due to computation time
    
    Inputs: krylov_dim -> Krylov dimension
            states (optional) -> str/ a list of index range
            Example (str) - only option: 'all'  (all Lanczos states)
            Example (list) - [3,large interger] : states from 3 to Krylov dim
    Outputs: from_state -> beginning state index
             to_state -> end state index'''

    if states == "all":
        from_state = 0
        to_state = krylov_dim
    else:
        from_state = states[0] 
        to_state = states[1] + 1

        if to_state > krylov_dim:
            to_state = krylov_dim
    
    return from_state, to_state

# ------- Function5: nonorthogonal overlap calculation/ collection --------
def overlap_nonortho_ref(cum_it,istate,Ylist,coeffs,refWF,path_nonortho_overlap):
    ''' Function to calculte/ upload overlap data of Lanczos states
    with  non-orthogonal refernce
    Inputs: cum_it -> cumulative iteration number
            istate -> Lanczos state to calculate overlap
            Ylist -> List of Krylov vectors
            coeffs -> Lanczos eigen coefficients
            refWF -> list of nonorthogonal references
            path_nonortho_overlap -> data file containing overlap 
            data (if available)
    Outputs: overlap -> overlap values as 1D numpy array
             overlap2 -> squared overalp values as 1D numpy array
             total -> total squared overlap'''

    num_ref = len(refWF)
    mstates = len(Ylist)
    if path_nonortho_overlap is None:
        overlap = np.zeros(num_ref,dtype=Ylist[0].rootNode.dtype)
        overlap2 = np.zeros(num_ref,dtype=Ylist[0].rootNode.dtype)
        total = np.zeros(num_ref, dtype=Ylist[0].rootNode.dtype)

        sum_overlap2 = np.array(0, dtype=Ylist[0].rootNode.dtype)
        for num in range(num_ref):
            for k in range(mstates):
                overlap[num] += bracket(refWF[num], Ylist[k])*coeffs[k,istate]
            overlap2[num] = overlap[num]**2
            sum_overlap2 += overlap2[num]
            total[num] = sum_overlap2

    elif path_nonortho_overlap is not None:
        filename = f"{path_nonortho_overlap}/Overlap_it{cum_it}_vec{istate}.out"
        overlap = np.loadtxt(filename,usecols=(3),skiprows=1,dtype=float)
        overlap2 = np.loadtxt(filename,usecols=(4),skiprows=1,dtype=float)
        
        num_ref = len(overlap)
        total = np.zeros(num_ref, dtype=overlap2[0].dtype)
        sum_overlap2 = np.array(0, dtype=Ylist[0].rootNode.dtype)
        for num in range(num_ref):
            sum_overlap2 += overlap2[num]
            total[num] = sum_overlap2 

    return overlap, overlap2, total

# --------- Main function: overlap, squared overlap and sum of squared overlap -----
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
            ref_dict -> dictionary containing REF info 
            path_nonortho_overlap (optional) -> nonorthogonal overlap file, if provided 
            then overlap is calculated using equation 2
    
    Generated file -> Overlap_it{i}_state{state}.out for iteration "i" and state "state"
    '''
    
    refE, refWF = collect_ref(ref_dict) 
    num_ref = len(refE)
    ref_in_cm = util.au2unit(np.array(refE),"cm-1")

    for it in range(start_cum,max_cum+1):
        Ylist = assemble_krylov_vectors(it,path_to_KS)
        eigenvalues, coeffs = eigen_info(it,path_to_KS)
        ev_in_cm = util.au2unit(np.array(eigenvalues),"cm-1")
        mstates = coeffs.shape[1]
        assert(mstates == len(Ylist))
   
        from_state, to_state = state_iterators(mstates,states_selected)
        for istate in range(from_state,to_state):
    
            overlap_file = open(f"Overlap_it{it}_vec{istate}.out","w")
            output_format = "{:>6} {:>16} {:>16} {:>16} {:>16} {:>16}" 
            lines = output_format.format("Index","RefE","eigenvalue","overlap","overlap-squared","Total\n")
            overlap_file.write(lines)

            overlap, overlap2, total = overlap_nonortho_ref(it,istate,Ylist,coeffs,refWF,path_nonortho_overlap)
            print(f"Note: Total squared overlap is {total[-1]})")
            num_ref = len(overlap) # some are truncated # ref_coeffs are sorted
            
            if ref_dict["ref_coeffs"] is not None:
                ref_coeffs = ref_dict["ref_coeffs"]
                is_real = np.isreal(ref_coeffs).all()
                overlapC = np.zeros(num_ref,dtype=Ylist[0].rootNode.dtype)
                overlap2C = np.zeros(num_ref,dtype=Ylist[0].rootNode.dtype)
                totalC = np.zeros(num_ref, dtype=Ylist[0].rootNode.dtype)

                sum_overlap2C = np.array(0, dtype=Ylist[0].rootNode.dtype)
                for num in range(num_ref):
                    for ic in range(num_ref):
                        if is_real:
                            overlapC[num] += ref_coeffs[ic,num] * overlap[ic]
                        else:
                            overlapC[num] += complex(ref_coeffs[ic,num]) * overlap[ic]
                    overlap2C[num] = overlapC[num]**2
                    sum_overlap2C += overlap2C[num]
                    totalC[num] = sum_overlap2C
                print(f"Note: Total squared ortho-overlap is {totalC[-1]})")

                overlap = overlapC
                overlap2 = overlap2C
                total = totalC

            for num in range(num_ref):
                lines = "{:>6}".format(num)
                lines += "{:>16}".format(f"{ref_in_cm[num]:.6f}")
                lines += "{:>16}".format(f"{ev_in_cm[istate]:.6f}")
                lines += "{:>16}".format(f"{overlap[num]:.4f}")
                lines += "{:>16}".format(f"{overlap2[num]:.4f}")
                lines += "{:>16}".format(f"{total[num]:.4f}")+"\n"
                overlap_file.write(lines)
            overlap_file.close()


# --------------------------- Main function ------------------------
if __name__ == "__main__":
    start_cum, max_cum = 1, 3 # cumulative iterations, same number for single iteration
    path_to_KS = "demo/saveTNSs/" # path to saved Krylov vectors
    states = 'all'     # Lanczos states; 'all' as string/a list of indices [0,2]; same number for single state
    
    path_to_ref = "/data/larsson/Eigen/RUNS/tns_D70/" # Path to References
    ref_energy = np.load(f"{path_to_ref}/matrices/evuv.npz")["ev"] # reference energies
    ref_coeffs = np.load(f"{path_to_ref}/matrices/evuv.npz")["uv"] # ortho coefficients
    ref_dict = {"path_to_ref":path_to_ref,"ref_energy":ref_energy,"ref_coeffs":ref_coeffs}
   
    calculate_and_write_overlap(start_cum,max_cum,path_to_KS,states,ref_dict,path_nonortho_overlap="check_overlap/nonortho/")
# ---------------  EOF ----------------------------------------------
