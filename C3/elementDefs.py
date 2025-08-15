#import constants as c #this ain't done to allow for different assumption sets
import pyfe3Dgcl as p3g
import numpy as np
import typing as ty
import copy
import pyfe3d.shellprop_utils as psu

'''Infinity Stiffness Spring'''
ks_mount = lambda INFTY_STIFF: p3g.SpringProp(INFTY_STIFF, INFTY_STIFF, INFTY_STIFF, INFTY_STIFF, INFTY_STIFF, INFTY_STIFF)

'''Battery Rail modelled as a beam'''
def prop_rail(din, E_STEEL, RHO_STEEL):
    prop_rail = p3g.OrientedBeamProp(1, 0, 0)
    prop_rail.E = E_STEEL
    prop_rail.A = np.pi/4*din**2
    prop_rail.Iyy = np.pi/64*din**4
    prop_rail.Izz = prop_rail.Iyy
    prop_rail.J = prop_rail.Iyy+prop_rail.Izz
    scf = 5/6 #ASSUMPTION, TODO: verify
    prop_rail.G = scf*E_STEEL/2/(1+0.3)
    prop_rail.intrho = RHO_STEEL*prop_rail.A
    prop_rail.intrhoy2 = RHO_STEEL*prop_rail.Izz
    prop_rail.intrhoz2 = RHO_STEEL*prop_rail.Iyy 
    return prop_rail

'''A template for a simpletruss beam elements'''
def prop_truss(rout, ft, E, rho):
    #NOTE: doesn't matter where - the truss is a tube hence symmetric, what matters is not aligned with the direction of any of the trusses
    assert (ft<=.1 and ft>0), f"The truss is not thin walled! Thickness/Radius: {ft}"
    prop_truss_ = p3g.OrientedBeamProp(1,1,1)
    prop_truss_.E = E
    t = rout*ft
    rm = rout-t/2
    prop_truss_.A = 2*np.pi*t*rm
    prop_truss_.Iyy = .5*np.pi*t*rm**3
    prop_truss_.Izz = prop_truss_.Iyy
    prop_truss_.J = prop_truss_.Iyy+prop_truss_.Izz
    scf = 5/6
    prop_truss_.G = scf*E/2/(1+.3)
    prop_truss_.intrho = rho*prop_truss_.A
    prop_truss_.intrhoy2 = rho*prop_truss_.Izz
    prop_truss_.intrhoz2 = rho*prop_truss_.Iyy
    return prop_truss_


'''Skin quad - for now'''
def skin_quad(E, nu, rho, t, h, sc, sb):
    "material direction"
    matx = 0 #TODO: maybe refine later - we do have some sweep
    maty = np.cos(np.radians(6))
    matz = np.sin(np.radians(6))

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

    return p3g.ShellPropAnddir(sp, matx, maty, matz)

def eledict(consts:ty.Dict[str,object], sizerVars:ty.Dict[str,object], codes:ty.Dict[str,str]):
    '''a function that initiates them all based on sizer and constant report'''

    eleProps = {"quad":{codes["skin"]:skin_quad(consts["E_ALU"], consts["NU"], consts["RHO_ALU"], sizerVars["tskin"],
                                                consts["ORTHG_H"], sizerVars["csp"], sizerVars["bsp"]),},
                "spring":{},
                "beam":{codes["rail"]:prop_rail(consts["DIN"], consts["E_STEEL"], consts["RHO_STEEL"]), 
                        codes["spar"]:prop_truss(consts["TRUSS_R"], sizerVars["fspar"], consts["E_ALU"], consts["RHO_ALU"]),
                        codes["LETE"]:prop_truss(consts["TRUSS_R"], sizerVars["fLETE"], consts["E_ALU"], consts["RHO_ALU"]),
                        codes["lg"]:prop_truss(consts["TRUSS_R"], sizerVars["flg"], consts["E_ALU"], consts["RHO_ALU"]),
                        codes["rib"]:prop_truss(consts["TRUSS_R"], sizerVars["frib"], consts["E_ALU"], consts["RHO_ALU"])}, 
                "mass":{}}
    
    return eleProps
