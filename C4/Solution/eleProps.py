from . import stressRecovery as sr

import typing as ty
import pyfe3d.beamprop as pbp
import pyfe3d.shellprop as psp
import pyfe3d.shellprop_utils as psu
import pyfe3d as pf3

def load_ele_props(desvars:ty.Dict[str,float], materials:ty.Dict[str,float], eleTypes:ty.List[str], eleArgs:ty.List[float])->ty.Dict[
    str,ty.List[object]]:
    '''
    loads current element properties in a format suitable for Pyfe3dModel updates
    
    :param desvars: current value of design variables
    :type desvars: ty.Dict[str, float]
    :param materials: a dictionary of material properties
    :type materials:Dict[string, float]
    :param eleTypes: list of element type codes
    :param eleTypes: List[str]
    :param eleArgs: arguments needed to initialise an element of a given type
    :type eleArgs: List[float]
    :return: element property lists to update Pyfe3dModel matrices with and to match element types to beams, quads etc.
    :rtype: Dict[str:List[object]]
    '''
    beamprops:ty.List[pbp.BeamProp] = list()
    beamorients:ty.List[ty.Tuple[float]] = list()
    shellprops:ty.List[psp.ShellProp] = list()
    matdirs:ty.List[ty.Tuple[float]] = list()
    inertia_vals:ty.List[ty.Tuple[float]] = list() #(m, Jxx, Jyy, Jzz) the inertia additions

    #NOTE: the eleTypes are separated for postprocessing, you don't need to postprocess inertia elements
    beamtypes:ty.List[str] = list()
    quadtypes:ty.List[str] = list()

    Ea = materials["E_ALU"]
    Ef = materials["E_FOAM"]
    nua = materials["NU_ALU"]
    nuf = materials["NU_FOAM"]
    rhoa = materials["RHO_ALU"]
    rhof = materials["RHO_FOAM"]

    for eleType, eleArg in zip(eleTypes, eleArgs):
        if eleType[1] == 'q': #sandwich elements
            H = eleArg[0]
            t = desvars[f"(2t/H)_{eleType}"]/2*H
            h = H-2*t
            shellprops.append(psu.laminated_plate([0,0,0], laminaprops=[(Ea, nua), (Ef, nuf), (Ea, nua)], 
                                                  plyts=[t, h, t], rhos=[rhoa, rhof, rhoa]))
            matdirs.append((0., 1., 0.))
        elif eleType[1] == 'i': #inertia elements
            inertia_vals.append(tuple(eleArg))
        elif eleType == 'rb': #rails -special case of a beam
            raise NotImplementedError
        elif eleType[1] == 'b': #other beams - rib elements
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid element ype code {eleType}!!!")
        
        #extracting beam and quad types into separate lists for postprocessing
        if eleType[1] == 'q':
            quadtypes.append(eleType)
        elif eleType[1] == 'b':
            beamtypes.append(eleType)
        
    return {'beamprops':beamprops,
            'beamorients':beamorients,
            'shellprops':shellprops,
            'matdirs':matdirs,
            'inertia_vals':inertia_vals,
            'beamtypes':beamtypes,
            'quadtypes':quadtypes}

def quad_stress_recovery(desvars:ty.Dict[str,float], materials:ty.Dict[str,float], quad:pf3.Quad4, shellprop:psp.ShellProp,
                          matdir:ty.Tuple[float], quadType:str, quadprobe:pf3.Quad4Probe)->float:
    '''
    Returns a failure margin for a quad element

    :param desvars: current value of design variables
    :type desvars: ty.Dict[str, float]
    :param materials: a dictionary of material properties
    :type materials:Dict[string, float]
    :param quad: the examined quad element
    :type quad: pf3.Quad4
    :param shellprop: the ShellProp of the examined quad
    :type shellprop: psp.ShellProp
    :param matdir: the material direction of the examined quad
    :type matdir: ty.Tuple[float]
    :param quadType: the type code of the examined quad
    :type quadType: str
    :param quadType: the probe the element is attached to
    :type quadType: Quad4Probe
    :return: the failure margin: ratio of criterion stress to allowable stress
    :rtype: float
    '''

    Ea = materials["E_ALU"]
    Ef = materials["E_FOAM"]
    nua = materials["NU_ALU"]
    nuf = materials["NU_FOAM"]
    sfa = materials["SF_ALU"]
    sff = materials["SF_FOAM"]

    #NOTE: we are looking for maximum stress in 4 places: on both the upper and lower boundaries of the core and on the outer sheet zs.
    strains = sr.strains_quad(quadprobe)
    normal_core, shear_core, tau_yz_core, tau_xz_core = sr.recover_stresses(strains, Ef, nuf, shellprop.scf_k13)
    normal_sheet, shear_sheet, tau_yz_sheet, tau_xz_sheet = sr.recover_stresses(strains, Ea, nua, shellprop.scf_k13)
    H = shellprop.h
    h = (1-desvars[f"(2t/H)_{quadType}"])*H #core height
    margins:ty.List[float] = list()

    #skin recovery
    for z in [-H/2, H/2]:
        s_xx, s_yy = normal_sheet(z)
        t_xy = shear_sheet(z)
        margins.append(sr.von_mises(s_xx, s_yy, 0., t_xy, tau_yz_sheet, tau_xz_sheet)/sfa)
    
    #core recovery
    for z in [-h/2, h/2]:
        s_xx, s_yy = normal_core(z)
        t_xy = shear_core(z)
        margins.append(sr.foam_crit(s_xx, s_yy, 0., t_xy, tau_yz_core, tau_xz_core, nuf)/sff)

    return max(margins)