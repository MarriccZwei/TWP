import pyfe3d as pf3
import aerosandbox.numpy as np 
import numpy.typing as nt
import typing as ty
import scipy.linalg as sl

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

    #NOTE: curvature signs - in the test when it bends up kappas are negative, we want negative strain on top if
    # bends up,so adding kxx and kyy seems right.
    normal_stresses = lambda z:E_pois*np.array([[1, nu], [nu, 1]])@np.array([kxx*z+exx, kyy*z+eyy])
    shear_stress = lambda z:G*(gxy+kxy*z)
    tau_yz = G*sc*gyz
    tau_xz = G*sc*gxz

    return normal_stresses, shear_stress, tau_yz, tau_xz

    
def von_mises(s_11, s_22, s_33, s_12, s_23, s_13):
    '''Von Mises failure criterion for a general stress tensor'''
    return np.sqrt(((s_11-s_22)**2+(s_22-s_33)**2+(s_33-s_11)**2)/2+3*(s_12**2+s_23**2+s_13**2))


def foam_crit(s_11, s_22, s_33, s_12, s_23, s_13, nu):
    '''Foam failure criterion as in Gioux et al. 2000. doi: https://doi.org/10.1016/S0020-7403(99)00043-0.
    With the change that regular poisson ratio, instead of the plastic one is used'''
    alpha2 = 9*(.5-nu)/(1+nu)
    s_princ = sl.eigh(np.array([(s_11, s_12, s_13), (s_12, s_22, s_23), (s_13, s_23, s_33)]), eigvals_only=True)
    s_v2 = ((s_princ[0]-s_princ[1])**2+(s_princ[1]-s_princ[2])**2+(s_princ[2]-s_princ[0])**2)/2
    s_m2 = np.average(s_princ)**2
    return np.sqrt((s_v2+alpha2*s_m2)/(1+alpha2/9))


def beam_strains(probe:pf3.Quad4Probe)->ty.Tuple[ty.Callable[[float, float], float]]:
    '''
    obtains the three meaningful strains for beam elements as functions of ye and ze
    
    :param probe: element probe with ue and xe already updated
    :type probe: pf3.Quad4Probe
    :return: the e_xx, e_xy, e_xz lambdas
    :rtype: Tuple[Callable[[float, float], float]]
    '''
    L = probe.xe[3]-probe.xe[0]
    epsilon = (probe.ue[6]-probe.ue[0])/L
    kappa_y = -(probe.ue[11]-probe.ue[5])/L
    kappa_z = (probe.ue[10]-probe.ue[4])/L
    gamma_y = (probe.ue[7]-probe.ue[1])/L-(probe.ue[11]+probe.ue[5])/2
    gamma_z = (probe.ue[8]-probe.ue[2])/L+(probe.ue[10]+probe.ue[4])/2
    kappa_x = (probe.ue[9]-probe.ue[3])/L

    e_xx = lambda ye, ze: epsilon+kappa_y*ye+kappa_z*ze
    e_xy = lambda ze: gamma_y-kappa_x*ze
    e_xz = lambda ye: gamma_z+kappa_x*ye
    return e_xx, e_xy, e_xz