import numpy as np
import numpy.typing as nt
import typing as ty
import scipy.sparse as ss
import scipy.sparse.linalg as ssl

'''INTERNAL FORCE UPDATE TODO: ADD STRAIN UPDATES'''
def update_after_displacement(meshOut:ty.Dict[str, object], sol:ty.Dict[str, object]):
    "Returns the KG matrix and internal force vector after the displacements from the solver are applied"
    KC0, M, N, x, y, z, mesh, up, ids, pts, nid_pos, ncoords = tuple(meshOut[k] for k in ['KC0', 'M', 'N', 'x', 'y', 'z', 'mesh', 'up', 'ids', 'pts', 'nid_pos', 'ncoords'])
    eleDict, KGv, KGr, KGc, ncoords_flatten = tuple(meshOut[k] for k in ['elements', 'KGv', 'KGr', 'KGc', 'ncoords_flatten'])
    fi = np.zeros(meshOut["N"])
    u, bu, bk, Kuu = tuple(sol[k] for k in ["u", "bu", "bk", "KC0uu"])


    for quad in eleDict["quad"]:
        # NOTE update affects the Quad4RProbe class attribute ue
        quad.update_probe_ue(u)
        quad.update_probe_xe(ncoords_flatten)
        quad.update_KG(KGr, KGc, KGv, quad.shellprop)
        quad.update_probe_finte(quad.shellprop)
        quad.update_fint(fi, quad.shellprop)
    
    for beam in eleDict["beam"]:
        beam.update_probe_ue(u)
        beam.update_probe_xe(ncoords_flatten)
        beam.update_KG(KGr, KGc, KGv, beam.beamprop, 0) #here only values change
        beam.update_probe_finte(beam.beamprop)
        beam.update_fint(fi, beam.beamprop)

    KG = ss.coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    KGuu = KG[bu, :][:, bu]

    return fi, KGuu, KG
    
'''GLOBAL ANALYSIS'''
def buckling(sol:ty.Dict[str, object], meshOut:ty.Dict[str, object], KGuu:nt.NDArray[np.float64]):
    K_KAuu = sol["KC0uu"]-sol["KAuu"]
    n_modes = 10 #NOTE: we only care about the load multiplier being smaller/greater than 1, but leave the k higher for accuracy
    eigenvects = np.zeros((meshOut["N"], n_modes), dtype=np.complex128)
    initial = 1
    done = False

    '''step 1) searching for the order of magnitude range'''
    ew, ev = ssl.eigs(A=K_KAuu, M=KGuu, k=n_modes, which="LM", sigma=initial)
    nim = np.count_nonzero(np.imag(ew))
    if nim==len(ew): #All results are non-real
        ew1 = ew
        ev1 = ev #placeholder values needed not to return a purely imaginary vector
        while nim==len(ew1):
            prev_initial = initial
            initial*=10
            ew1, ev1 = ssl.eigs(A=K_KAuu, M=KGuu, k=n_modes, which="LM", sigma=initial)
            nim = np.count_nonzero(np.imag(ew1))
            print(f"nim: {nim}")
        ew = ew1 #we only update eigpairs if we know there are any real ones in the range
        ev = ev1
        lower_bound = prev_initial
        upper_bound = initial #since we are ascending
    else: #not all results are nonreal
        while nim<len(ew):
            prev_initial = initial
            initial/=10
            ew1, ev1 = ssl.eigs(A=K_KAuu, M=KGuu, k=n_modes, which="LM", sigma=initial)
            nim = np.count_nonzero(np.imag(ew1)) #again not to leave afully complex output
            print(f"nim: {nim}")
        lower_bound = initial
        upper_bound = prev_initial #since we are descending
    

    '''step 2) if we found our region'''
    for i in range(5): #three iterations should be enough for convergence
        print(f"lb: {lower_bound}, ub: {upper_bound}, nim: {nim}")
        sgm = (lower_bound+upper_bound)/2
        ew1, ev1 = ssl.eigs(A=K_KAuu, M=KGuu, k=n_modes, which="LM", sigma=sgm)
        nim = np.count_nonzero(np.imag(ew1))
        if nim==len(ew1): #We landed in a fully complex region
            lower_bound = sgm
        else: #we are in the complex region, we can go up
            upper_bound = sgm
            ew = ew1 #we only update eigpairs if we know there are any real ones in the range
            ev = ev1

    eigenvects[sol["bu"], :] = ev

    #reformatting the result as load multiplier, eigenvalues and an array of eigenvectors
    return min(ew[np.isclose(np.imag(ew), 0)]), ew, eigenvects.T

def natfreq(sol:ty.Dict[str, object], meshOut:ty.Dict[str, object]):
    K_KAuu = sol["KC0uu"]-sol["KAuu"]
    Muu = meshOut["M"][sol["bu"], :][:, sol["bu"]]
    n_modes=7#TODO: How many to use
    eigenvects = np.zeros((meshOut["N"], n_modes), dtype=np.complex128)

    ew, ev = ssl.eigs(A=K_KAuu, M=Muu, k=n_modes, which="LM", sigma=-1.) #TODO: copy pasted. verify it still holds
    eigenvects[sol["bu"], :] = ev

    #reformatting the result as omega_ns and an array of eigenvectors
    return ew**.5, eigenvects.T

'''LOCAL ANALYSIS - LOCAL BUCKLING + ELEMENT STRESSES'''
class BeamPlotData():
    def __init__(self, node1, node2, tau, buckl, sgm):
        self.node1 = node1
        self.node2 = node2
        self.sgm = sgm
        self.tau = tau
        self.buc = buckl

class QuadPlotData():
    def __init__(self, node1, node2, node3, node4, tau, sgm):
        self.node1 = node1
        self.node2 = node2
        self.node3 = node3
        self.node4 = node4
        self.tau = tau
        self.sgm = sgm

def beam_stresses(meshOut:ty.Dict[str, object], sizerVars:ty.Dict[str, object], csts:ty.Dict[str, object]):
    '''TODO: consistent stress computation from strains'''
    buckl_ratio_max = 0
    tau_ratio_max = 0
    sigma_ratio_max = 0
    beams2plot:ty.List[BeamPlotData] = list()
    nid_pos = meshOut["nid_pos"]
    for ele in meshOut["elements"]["beam"]:
        beams2plot.append(BeamPlotData(nid_pos[ele.n1], nid_pos[ele.n2], 0, 0, 0))
    return buckl_ratio_max, tau_ratio_max, sigma_ratio_max, beams2plot

def quad_stresses(meshOut:ty.Dict[str, object], sizerVars:ty.Dict[str, object], csts:ty.Dict[str, object]):
    '''TODO: consistent stress computation from strains'''
    tau_ratio_max = 0
    sigma_ratio_max = 0
    quads2plot:ty.List[QuadPlotData] = list()
    nid_pos = meshOut["nid_pos"]
    for ele in meshOut["elements"]["quad"]:
        quads2plot.append(QuadPlotData(nid_pos[ele.n1], nid_pos[ele.n2], nid_pos[ele.n3], nid_pos[ele.n4], 0, 0))
    return tau_ratio_max, sigma_ratio_max, quads2plot