import pyfe3d as pf3
import aerosandbox.numpy as np 
import numpy.typing as nt
import typing as ty
import scipy.linalg as sl

def strains_quad(probe:pf3.Quad4RProbe):
    '''
    Outputs the strains vector: [exx, eyy, gxy, kxx, kyy, kxy, gyz, gxz]
    Obtained by constructing the BL matrix and multiplying it with the displacements vector
    BLgxz, BLgyz are obtained by summing up the rot and grad BL row vectors before multiplication
    BL constructed at xi, eta = (0,0), the element's integratio point
    NOTE: The strains follow pyfe3d naming and sign convention, not the stress tensor one!
    '''
    
    # NOTE adapted from Quad4R source code
    x1 = probe.xe[0]
    y1 = probe.xe[1]
    x2 = probe.xe[3]
    y2 = probe.xe[4]
    x3 = probe.xe[6]
    y3 = probe.xe[7]
    x4 = probe.xe[9]
    y4 = probe.xe[10]

    j11 = 2.0*(-y1 - y2 + y3 + y4)/(x1*y2 - x1*y4 - x2*y1 + x2*y3 - x3*y2 + x3*y4 + x4*y1 - x4*y3)
    j12 = 2.0*(y1 - y2 - y3 + y4)/(x1*y2 - x1*y4 - x2*y1 + x2*y3 - x3*y2 + x3*y4 + x4*y1 - x4*y3)
    j21 = 2.0*(x1 + x2 - x3 - x4)/(x1*y2 - x1*y4 - x2*y1 + x2*y3 - x3*y2 + x3*y4 + x4*y1 - x4*y3)
    j22 = 2.0*(-x1 + x2 + x3 - x4)/(x1*y2 - x1*y4 - x2*y1 + x2*y3 - x3*y2 + x3*y4 + x4*y1 - x4*y3)

    N1 = 0.25
    N2 = 0.25
    N3 = 0.25
    N4 = 0.25

    N1x = -0.25*j11 - 0.25*j12
    N2x = 0.25*j11 - 0.25*j12
    N3x = 0.25*j11 + 0.25*j12
    N4x = -0.25*j11 + 0.25*j12
    N1y = -0.25*j21 - 0.25*j22
    N2y = 0.25*j21 - 0.25*j22
    N3y = 0.25*j21 + 0.25*j22
    N4y = -0.25*j21 + 0.25*j22

    #NOTE adapted from Quad4Probe source code for constructing the strain interpolation matrix BL
    BL_LEN = 4*pf3.DOF
    BLexx = np.zeros(BL_LEN, dtype=np.float64)
    BLeyy = np.zeros(BL_LEN, dtype=np.float64)
    BLgxy = np.zeros(BL_LEN, dtype=np.float64)

    BLkxx = np.zeros(BL_LEN, dtype=np.float64)
    BLkyy = np.zeros(BL_LEN, dtype=np.float64)
    BLkxy = np.zeros(BL_LEN, dtype=np.float64)

    BLgyz_grad = np.zeros(BL_LEN, dtype=np.float64)
    BLgyz_rot = np.zeros(BL_LEN, dtype=np.float64)
    BLgxz_grad = np.zeros(BL_LEN, dtype=np.float64)
    BLgxz_rot = np.zeros(BL_LEN, dtype=np.float64)

    BLexx[0] = N1x
    BLexx[6] = N2x
    BLexx[12] = N3x
    BLexx[18] = N4x

    BLeyy[1] = N1y
    BLeyy[7] = N2y
    BLeyy[13] = N3y
    BLeyy[19] = N4y

    BLgxy[0] = N1y
    BLgxy[6] = N2y
    BLgxy[12] = N3y
    BLgxy[18] = N4y
    BLgxy[1] = N1x
    BLgxy[7] = N2x
    BLgxy[13] = N3x
    BLgxy[19] = N4x

    BLkxx[4] = N1x
    BLkxx[10] = N2x
    BLkxx[16] = N3x
    BLkxx[22] = N4x

    BLkyy[3] = -N1y
    BLkyy[9] = -N2y
    BLkyy[15] = -N3y
    BLkyy[21] = -N4y

    BLkxy[3] = -N1x
    BLkxy[9] = -N2x
    BLkxy[15] = -N3x
    BLkxy[21] = -N4x
    BLkxy[4] = N1y
    BLkxy[10] = N2y
    BLkxy[16] = N3y
    BLkxy[22] = N4y

    BLgyz_grad[2] = N1y
    BLgyz_grad[8] = N2y
    BLgyz_grad[14] = N3y
    BLgyz_grad[20] = N4y

    BLgyz_rot[3] = -N1
    BLgyz_rot[9] = -N2
    BLgyz_rot[15] = -N3
    BLgyz_rot[21] = -N4

    BLgxz_grad[2] = N1x
    BLgxz_grad[8] = N2x
    BLgxz_grad[14] = N3x
    BLgxz_grad[20] = N4x

    BLgxz_rot[4] = N1
    BLgxz_rot[10] = N2
    BLgxz_rot[16] = N3
    BLgxz_rot[22] = N4

    return np.vstack([np.asarray(BLexx),
                      np.asarray(BLeyy),
                      np.asarray(BLgxy),
                      np.asarray(BLkxx),
                      np.asarray(BLkyy),
                      np.asarray(BLkxy),
                      np.asarray(BLgyz_rot)+np.asarray(BLgyz_grad),
                      np.asarray(BLgxz_rot)+np.asarray(BLgxz_grad),])@np.asarray(probe.ue)


def recover_stresses(strains:nt.NDArray[np.float32], E:float, nu:float, sc:float):
    '''Stress recovery for an isotropic plate
    
    :param strains: strains as provided by strains_quad function
    :type strains: NDArray[np.float32]
    :param E: material's Elastic modulus
    :type E: float
    :param nu: material's Poisson's ratio modulus
    :type nu: float
    :param sc: transverse shear correction factor
    :type sc: float

    :return: the normall stresses lmbda, the in-plane shear stress lambda, out of plane shear stresses
    :rtype: Tuple[Callable[[float],  NDArray[np.float32]], Callable[[float],  float], float, float]
    '''

    G = E/2/(1+nu)
    E_pois = E/(1-nu**2)
    exx = strains[0]
    eyy = strains[1]
    gxy = strains[2]
    kxx = strains[3]
    kyy = strains[4]
    kxy = -strains[5] #so that moment that translates to positive shear on top actually causes positive shear on top. see tests
    #to satisfy the sign convention of the stress tensor of positive xz shear causing shrinking along x=z line, 
    #the yz in pyfe3d strain is actually the xz one in the stress tensor as well. See tests.
    gyz = -strains[7]
    gxz = -strains[6]

    #NOTE: curvature signs - in the test when it bends up kappas are negative, we want negative strain on top if
    # bends up,so adding kxx and kyy seems right.
    normal_stresses = lambda z:E_pois*np.array([[1, nu], [nu, 1]])@np.array([kxx*z+exx, kyy*z+eyy]) #kxx is bending along y, actually
    shear_stress = lambda z:G*(gxy+kxy*z)
    tau_yz = G*sc*gyz
    tau_xz = G*sc*gxz

    return normal_stresses, shear_stress, tau_xz, tau_yz 

    
def von_mises(s_11, s_22, s_33, s_12, s_23, s_13):
    '''Von Mises failure criterion for a general stress tensor
    
    s_11, s_22, s_33 are normal stresses, s_12, s_23, s_13 are shear stresses'''
    return np.sqrt(((s_11-s_22)**2+(s_22-s_33)**2+(s_33-s_11)**2)/2+3*(s_12**2+s_23**2+s_13**2))


def foam_crit(s_11, s_22, s_33, s_12, s_23, s_13, nu):
    '''Foam failure criterion as in Gioux et al. 2000. doi: https://doi.org/10.1016/S0020-7403(99)00043-0.
    With the change that regular poisson ratio, instead of the plastic one is used
    
    s_11, s_22, s_33 are normal stresses, s_12, s_23, s_13 are shear stresses, nu is poisson ratio'''
    alpha2 = 9*(.5-nu)/(1+nu)
    s_princ = sl.eigh(np.array([(s_11, s_12, s_13), (s_12, s_22, s_23), (s_13, s_23, s_33)]), eigvals_only=True)
    s_v2 = ((s_princ[0]-s_princ[1])**2+(s_princ[1]-s_princ[2])**2+(s_princ[2]-s_princ[0])**2)/2
    s_m2 = np.average(s_princ)**2
    return np.sqrt((s_v2+alpha2*s_m2)/(1+alpha2/9))


def beam_strains(probe:pf3.Quad4RProbe)->ty.Tuple[ty.Callable[[float, float], float]]:
    '''
    obtains the three meaningful strains for beam elements as functions of ye and ze
    
    :param probe: element probe with ue and xe already updated
    :type probe: pf3.Quad4RProbe

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