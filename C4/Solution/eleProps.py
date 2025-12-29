import typing as ty
import pyfe3d.beamprop as pbp
import pyfe3d.shellprop as psp
import pyfe3d.shellprop_utils as psu

def load_ele_props(desvars:ty.Dict[str,float], materials:ty.Dict[str,float], eleTypes:ty.List[str], eleArgs:ty.List[float])->ty.Tuple[ty.List[pbp.BeamProp], 
    ty.List[ty.Tuple[float]], ty.List[psp.ShellProp], ty.List[ty.Tuple[float]], ty.List[ty.Tuple[float]]]:
    '''
    loads current element properties in a format suitable for Pyfe3dModel updates
    
    :param desvars: current value of design variables
    :type desvars: ty.Dict[str, float]
    :param eleTypes: list of element type codes
    :param eleTypes: List[str]
    :param eleArgs: arguments needed to initialise an element of a given type
    :type eleArgs: List[float]
    :return: element property list to update Pyfe3dModel matrices with
    :rtype: Tuple[List, List[Tuple[float]], List, List[Tuple[float]], List[Tuple[float]]]
    '''
    beamprops:ty.List[pbp.BeamProp] = list()
    beamorients:ty.List[ty.Tuple[float]] = list()
    shellprops:ty.List[psp.ShellProp] = list()
    matdirs:ty.List[ty.Tuple[float]] = list()
    inertia_vals:ty.List[ty.Tuple[float]] = list() #(m, Jxx, Jyy, Jzz) the inertia additions

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
            shellprops.append(psu.laminated_plate([0,0,0], laminaprops=[(Ea, nua), (Ef, nuf), (Ea, nua)], plyts=[t, h, t], rhos=[rhoa, rhof, rhoa]))
            matdirs.append((0., 1., 0.))
        elif eleType[1] == 'i': #inertia elements
            inertia_vals.append(tuple(eleArg))
        elif eleType == 'rb': #rails -special case of a beam
            raise NotImplementedError
        elif eleType[1] == 'b': #other beams - rib elements
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid element ype code {eleType}!!!")
        
    return beamprops, beamorients, shellprops, matdirs, inertia_vals