import typing as ty
import pyfe3d.beamprop as pbp
import pyfe3d.shellprop as psp

def load_ele_props(desvars:ty.Dict[str,float], eleTypes:ty.List[str], eleArgs:ty.List[float])->ty.Tuple[ty.List[pbp.BeamProp], 
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

    for eleType, eleArg in zip(eleTypes, eleArgs):
        if eleType == "sq":
            '''etc. Append the right lists'''
