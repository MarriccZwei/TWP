import pyfe3d as pf3
import numpy as np 
import numpy.typing as nt
import typing as ty
import scipy.linalg as sl
import pyfe3d.shellprop as psp

def strains_quad(probe:pf3.Quad4Probe):
    '''
    Outputs the strains vector: [exx, eyy, gxy, kxx, kyy, kxy, gyz, gxz]
    Obtained by constructing the BL matrix and multiplying it with the displacements vector
    BLgxz, BLgyz are obtained by summing up the rot and grad BL row vectors before multiplication
    BL constructed at xi, eta = (0,0), the element's integratio point
    '''
    probe.update_BL(0,0)
    return np.vstack([np.asarray(probe.BLexx),
                      np.asarray(probe.BLeyy),
                      np.asarray(probe.BLgxy),
                      np.asarray(probe.BLkxx),
                      np.asarray(probe.BLkyy),
                      np.asarray(probe.BLkxy),
                      np.asarray(probe.BLgyz_rot)+np.asarray(probe.BLgyz_grad),
                      np.asarray(probe.BLgxz_rot)+np.asarray(probe.BLgxz_grad),])@np.asarray(probe.ue)

def get_ABDE(prop:psp.ShellProp):
    return np.array([[prop.A11, prop.A12, prop.A16, prop.B11, prop.B12, prop.B16, 0, 0],
                         [prop.A12, prop.A22, prop.A26, prop.B12, prop.B22, prop.B26, 0, 0],
                         [prop.A16, prop.A26, prop.A66, prop.B16, prop.B26, prop.B66, 0, 0],
                         [prop.B11, prop.B12, prop.B16, prop.D11, prop.D12, prop.D16, 0, 0],
                         [prop.B12, prop.B22, prop.B26, prop.D12, prop.D22, prop.D26, 0, 0],
                         [prop.B16, prop.B26, prop.B66, prop.D16, prop.D26, prop.D66, 0, 0],
                         [0, 0, 0, 0, 0, 0, prop.E44, prop.E45],
                         [0, 0, 0, 0, 0, 0, prop.E45, prop.E55]], dtype=pf3.DOUBLE)

def recover_stresses(strains:nt.NDArray[np.float32], E:float, nu:float, sc:float):
    '''Stress recovery for an isotropic plate'''

    G = E/2/(1+nu)
    E_pois = E/(1-nu**2)
    exx = strains[0]
    eyy = strains[1]
    gxy = strains[2]
    kxx = strains[3]
    kyy = strains[4]
    kxy = strains[5]
    gyz = strains[6]
    gxz = strains[7]

    #TODO: curvature signs - in the test when it bends up kappas are negative, we want negative strain on top if
    # bends up,so adding kxx and kyy seems right.Ultimately it also does not matter because the sandwich is symmetric
    normal_stresses = lambda z:E_pois*np.array([[1, nu], [nu, 1]])@np.array([kxx*z+exx, kyy*z+eyy])
    shear_stress = lambda z:G*(gxy+kxy*z) #same story sign does not matter as you would pythagoras this anyway
    tau_yz = G*sc*gyz
    tau_xz = G*sc*gxz

    return normal_stresses, shear_stress, tau_yz, tau_xz

#TODO: get rid of
def tripple_mohr(normal_stresses:ty.Callable, shear_stress:ty.Callable, tau_yz:float, tau_xz:float, zmax:float) -> float:
    '''
    Computes the highest mohr circle radius for a set of strains for a layer within a plate
    '''
    mohr_R = lambda sx, sy, txy: (((sx-sy)/2)**2+txy**2)**.5
    sx_zmin, sy_zmin = normal_stresses(-zmax)
    sx_zmax, sy_zmax = normal_stresses(zmax)
    txy_min = shear_stress(-zmax)
    txy_max = shear_stress(zmax)
    return max(mohr_R(sx_zmin, sy_zmin, txy_min),
               mohr_R(sx_zmax, sy_zmax, txy_max),
               mohr_R(sx_zmin, 0, tau_yz),
               mohr_R(sx_zmax, 0, tau_yz),
               mohr_R(sy_zmin, 0, tau_xz),
               mohr_R(sy_zmax, 0, tau_xz))


def sandwich_recovery(probe:pf3.Quad4Probe, zmaxs:ty.List[float], Es:ty.List[float], nus:ty.List[float], sc:float) -> ty.List[float]:
    '''
    Computes the mohr radii for each of the isotropic layers of a symmetric sandwich
    '''
    strains = strains_quad(probe)
    Rs = list()
    
    for zmax, E, nu in zip(zmaxs, Es, nus):
        normal_core, shear_core, tyz_core, txz_core = recover_stresses(strains, E, nu, sc)
        Rs.append(tripple_mohr(normal_core, shear_core, tyz_core, txz_core, zmax))

    return Rs
#=====================================
    
def von_mises(s_11, s_22, s_33, s_12, s_23, s_13):
    return np.sqrt(((s_11-s_22)**2+(s_22-s_33)**2+(s_33-s_11)**2)/2+3*(s_12**2+s_23**2+s_13**2))

def foam_crit(s_11, s_22, s_33, s_12, s_23, s_13, nu):
    alpha2 = 9*(.5-nu)/(1+nu)
    s_princ = sl.eigh(np.array([(s_11, s_12, s_13), (s_12, s_22, s_23), (s_13, s_23, s_33)]), eigvals_only=True)
    s_v2 = ((s_princ[0]-s_princ[1])**2+(s_princ[1]-s_princ[2])**2+(s_princ[2]-s_princ[0])**2)/2
    s_m2 = np.average(s_princ)**2
    return np.sqrt((s_v2+alpha2*s_m2)/(1+alpha2/9))