import pyfe3d as pf3
import numpy as np 
import numpy.typing as nt
import typing as ty
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


def tripple_mohr(normal_stresses:ty.Callable, shear_stress:ty.Callable, tau_yz:float, tau_xz:float, zc:float, zs:float) -> ty.Tuple[float]:
    '''
    Computes the highest mohr circle radius for a set of strains
    '''
    