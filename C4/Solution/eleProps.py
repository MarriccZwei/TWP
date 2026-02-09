from . import stressRecovery as sr

import typing as ty
import pyfe3d.beamprop as pbp
import pyfe3d.shellprop as psp
import pyfe3d.shellprop_utils as psu
import pyfe3d as pf3
import aerosandbox.numpy as np

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

    #Timoshenko 1922 https://doi.org/10.1080/14786442208633855
    scfr = (5+5*nua)/(6+5*nua)
    Gr = scfr*Ea/2/(1+nua)
    scfc = (6+12*nua+6*nua**2)/(7+12*nua+4*nua**2)
    Gc = scfc*Ea/2/(1+nua)

    for eleType, eleArg in zip(eleTypes, eleArgs):
        if eleType[1] == 'q': #sandwich elements
            H = eleArg[0]#*desvars[f"(H'/H)_{eleType}"] NOTE: still disputed - probs 2 remove
            t = desvars[f"(2t/H)_{eleType}"]/2*H
            h = H-2*t
            shellprops.append(psu.laminated_plate([0,0,0], laminaprops=[(Ea, nua), (Ef, nuf), (Ea, nua)], 
                                                  plyts=[t, h, t], rhos=[rhoa, rhof, rhoa]))
            matdirs.append((0., 1., 0.))

        elif eleType[1] == 'i': #inertia elements #NOTE 4 now all ass hard-coded cylinders
            mi = eleArg[0]
            Jxx = 0.
            Jyy = 0.
            Jzz = 0.
            if eleType[0] == 'b':
                rhob = 3000
                lb = 18/15
                rb = np.sqrt(mi/rhob/lb/np.pi)
                Jxx = mi*(lb**2/12+rb**2/4)
                Jyy = mi*rb**2/2
                Jzz = Jxx
            elif eleType[0] == 'l':
                llg = 2.77732
                rlg = .801187
                Jxx = mi*rlg**2/2
                Jyy = mi*(rlg**2/4+llg**2/12)
                Jzz = Jyy
            elif eleType[0] == 'm':
                lm = 2.61709
                Jyy = mi*lm**2/12
                Jzz = mi*lm**2/12
            inertia_vals.append((eleArg[0], Jxx, Jyy, Jzz))

        elif eleType in ['rb', 'sb']: #rails & scaffolding - circular beams
            D = eleArg[0] if eleType=='rb' else desvars["Ds"]
            prop_rail = pbp.BeamProp()
            prop_rail.E = Ea
            prop_rail.A = np.pi/4*D**2
            prop_rail.Iyy = np.pi/64*D**4
            prop_rail.Izz = prop_rail.Iyy
            prop_rail.J = prop_rail.Iyy+prop_rail.Izz
            prop_rail.G = Gc
            prop_rail.intrho = rhoa*prop_rail.A
            prop_rail.intrhoy2 = rhoa*prop_rail.Izz
            prop_rail.intrhoz2 = rhoa*prop_rail.Iyy
            beamprops.append(prop_rail)
            beamorients.append((1.,0.,0.))

        elif eleType[1] == 'b': #other beams - rib elements
            W = desvars[f"W_{eleType}"]
            H = eleArg[0]
            prop_truss_ = pbp.BeamProp()
            prop_truss_.E = Ea
            prop_truss_.A = W*H
            prop_truss_.Iyy = W*H**3/12
            prop_truss_.Izz = W**3*H/12
            prop_truss_.J = prop_truss_.Iyy+prop_truss_.Izz
            prop_truss_.G = Gr
            prop_truss_.intrho = rhoa*prop_truss_.A
            prop_truss_.intrhoy2 = rhoa*prop_truss_.Izz
            prop_truss_.intrhoz2 = rhoa*prop_truss_.Iyy
            beamprops.append(prop_truss_)
            beamorients.append((0.,1.,0.))

        elif eleType[1] == 's':
            pass #springs have their properties inside them
        else:
            raise ValueError(f"Invalid element code for a element {eleType}!!!")
        
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


def beam_stress_recovery(desvars:ty.Dict[str,float], materials:ty.Dict[str,float], beam:pf3.BeamC, beamprop:pbp.BeamProp,
                          beamorient:ty.Tuple[float], beamType:str, beamprobe:pf3.BeamCProbe)->float:
    '''
    Returns a failure margin for a beam element
    
    :param desvars: current value of design variables
    :type desvars: ty.Dict[str, float]
    :param materials: a dictionary of material properties
    :type materials:Dict[string, float]
    :param beam: The examined beam element
    :type beam: pf3.BeamC
    :param beamprop: BeamProp belonging to the element
    :type beamprop: pbp.BeamProp
    :param beamorient: beam orientation belonging to the elemnt
    :type beamorients: ty.Tuple[float]
    :param beamType: beam element type code
    :type beamType: str
    :param beamprobe: an updated probe set to the element
    :type beamprobe: pf3.BeamCProbe
    :return: Ratio of stress experienced by the beam element to the failure stress
    :rtype: float
    '''
    E = beamprop.E
    G = beamprop.G
    sfa = materials["SF_ALU"]

    if beamType[1]=='b':
        exx, exy, exz = sr.beam_strains(beamprobe)
    else:
        raise ValueError(f"Expected, beam elements, got a non beam: {beamType}!!!")
    
    #cross section points to evaluate
    yes = list()
    zes = list()

    if beamType in ['rb', 'sb']: #rail or scaffold evaluation
        R = np.sqrt(beamprop.A/np.pi)
        EVAL_PTS = 16 #how many points to evaluate around cross-section
        theta = np.linspace(0, np.pi*2, EVAL_PTS+1)[:-1]
        yes = R*np.sin(theta)
        zes = R*np.cos(theta)
        
    else: #rib evaluation
        W = desvars[f"W_{beamType}"]
        H = beamprop.A/W
        Wper2 = W/2
        Hper2 = H/2
        yes = [Wper2, -Wper2, -Wper2, Wper2]
        zes = [Hper2, Hper2, -Hper2, -Hper2]

    #standard von mises evaluation procedure at selected cross section points
    s_vmises = list()
    for ye, ze in zip(yes, zes):
        sigma = exx(ye, ze)*E
        tau_xy = exy(ze)*G
        tau_xz = exz(ye)*G
        s_vmises.append(sr.von_mises(sigma, 0., 0., tau_xy, tau_xz, 0.))
    return max(s_vmises)/sfa