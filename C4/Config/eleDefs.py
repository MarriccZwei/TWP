import pyfe3d.beamprop as pbp
import pyfe3d.shellprop as psp
import pyfe3d.shellprop_utils as psu
import typing as ty
import numpy as np

from .Config import Config

'''Element definitions that taken arguments and config, return beamprops and shellprops'''
def rail_beam(config:Config) -> ty.Tuple[pbp.BeamProp, ty.Tuple[float]]:
    """
    Rail elements of constant diameter modelled as filled steel circular cross-section beams
    """
    orient = (1, 0, 0)
    prop_rail = pbp.BeamProp
    prop_rail.E = config.E_STEEL
    prop_rail.A = np.pi/4*config.DIN**2
    prop_rail.Iyy = np.pi/64*config.DIN**4
    prop_rail.Izz = prop_rail.Iyy
    prop_rail.J = prop_rail.Iyy+prop_rail.Izz
    scf = 5/6 #ASSUMPTION, TODO: verify
    prop_rail.G = scf*config.E_STEEL/2/(1+0.3)
    prop_rail.intrho = config.RHO_STEEL*prop_rail.A
    prop_rail.intrhoy2 = config.RHO_STEEL*prop_rail.Izz
    prop_rail.intrhoz2 = config.RHO_STEEL*prop_rail.Iyy 
    return prop_rail, orient

def tubular_beam(chord:float, config:Config, ft:float) -> ty.Tuple[pbp.BeamProp, ty.Tuple[float]]:
    """
    A function for all the truss beams, i.e. aluminium diameter proprtional to chord, hollow,
    but not necessarily thin-walled.

    chord - aircraft chord [m], passed as an input to truss functions;
    ft - thickness to radius ratio [-], constant for any type of tube.
    """
    
    assert (ft<=.9 and ft>0), f"The truss is not hollow! Thickness/Radius: {ft}"
    rout = config.TRUSS_RC*chord
    E = config.E_ALU
    rho = config.RHO_ALU

    #NOTE: doesn't matter where - the truss is a tube hence symmetric, what matters is not aligned with the direction of any of the trusses
    orient = (1,1,1)

    prop_truss_ = pbp.BeamProp()
    prop_truss_.E = E
    t = rout*ft
    rin = rout-t
    prop_truss_.A = np.pi*rout**2-np.pi*rin**2
    prop_truss_.Iyy = .5*np.pi*(rout**4-rin**2)
    prop_truss_.Izz = prop_truss_.Iyy
    prop_truss_.J = prop_truss_.Iyy+prop_truss_.Izz
    scf = 5/6
    prop_truss_.G = scf*E/2/(1+.3)
    prop_truss_.intrho = rho*prop_truss_.A
    prop_truss_.intrhoy2 = rho*prop_truss_.Izz
    prop_truss_.intrhoz2 = rho*prop_truss_.Iyy
    return prop_truss_, orient

def skin_quad(config:Config, thickness:float, spacing_b:float, spacing_c:float)->ty.Tuple[psp.ShellProp, ty.Tuple[float]]:
    """
    A returns a shellprop and material direction (direction of the spanwise stiffeners,
    which are spaced chordwise so their spacing is denoted as spacing_c), spacing_b is the spanwise 
    spacing of the chordwise stiffeners. thickness is the thickness of bothe the skin and the
    stiffeners, is [m]
    """
    E = config.E_ALU
    h = config.ORTHG_H
    t = thickness
    sc = spacing_c
    sb = spacing_b
    nu = config.NU
    rho = config.RHO_ALU

    "material#TODO: maybe refine later - we do have some sweep direction"
    matx = 0 
    maty = np.cos(np.radians(6))
    matz = np.sin(np.radians(6))
    matdir = (matx, maty, matz)

    sp = psu.isotropic_plate(t, E, nu, rho=rho)
    scf = 5/6 #NOTE: Default assumtion
    G = scf*E/2/(1+.3) #TODO: maybe later use an explicit G
    "smeared stiffener matrix entries"
    #A matrix
    sp.A11+=E*h*t/sc
    sp.A22+=E*h*t/sb
    #B matrix
    sp.B11+=E*h*t*(h+t)/2/sc
    sp.B22+=E*h*t*(h+t)/2/sb
    #D matrix
    sp.D11+=E*h*t/4*sc*(h**2/3+(h+t)**2)
    sp.D22+=E*h*t/4*sb*(h**2/3+(h+t)**2)
    sp.D66+=G*h**3*t/24*(1/sc+1/sb)
    #NOTE: neglecting the E terms for now, assume the prop sufficiently thin

    return sp, matdir