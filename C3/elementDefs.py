#import constants as c #this ain't done to allow for different assumption sets
import pyfe3Dgcl as p3g
import numpy as np
import typing as ty
import copy
import pyfe3d.shellprop_utils as psp

'''Infinity Stiffness Spring'''
ks_mount = lambda INFTY_STIFF: p3g.SpringProp(INFTY_STIFF, INFTY_STIFF, INFTY_STIFF, INFTY_STIFF, INFTY_STIFF, INFTY_STIFF)

'''Battery Rail modelled as a beam'''
def prop_rail(din, E_STEEL, RHO_STEEL):
    prop_rail = p3g.OrientedBeamProp(1, 0, 0)
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

'''Flange of Panel Ribs modelled as beam'''
def rib_flange(t_rib, ribflange, E, RHO):
    #template for the flange prop
    prop_flange = p3g.OrientedBeamProp(0, 0, 0)
    prop_flange.A = t_rib*ribflange
    prop_flange.Izz = t_rib*ribflange**3/12
    prop_flange.Iyy = ribflange*t_rib**3/12
    scf = 5/6 #ASSUMPTION, TODO: verify
    prop_flange.G = scf*E/2/(1+0.3)
    prop_flange.intrho = RHO*prop_flange.A
    prop_flange.intrhoy2 = RHO*prop_flange.Izz
    prop_flange.intrhoz2 = RHO*prop_flange.Iyy

    def prop_flange_or(dir): #oriented flange prop
        pfor = copy.deepcopy(prop_flange)
        pfor.xyex = dir[0]
        pfor.xyey = dir[1]
        pfor.xyez = dir[2]
        return pfor
    return prop_flange_or

'''Spar quad'''
spar_quad = lambda t_spar, E, NU, RHO: psp.isotropic_plate(thickness=t_spar, E=E, nu=NU, rho=RHO, calc_scf=True)

'''Skin quad'''
skin_quad = lambda t_skin, E, NU, RHO:psp.isotropic_plate(thickness=t_skin, E=E, nu=NU, rho=RHO, calc_scf=True)

'''Panel Plate quad'''
panel_plate_quad = lambda t_plate, E, NU, RHO:psp.isotropic_plate(thickness=t_plate, E=E, nu=NU, rho=RHO, calc_scf=True)

'''Panel Rib quad'''
panel_rib_quad = lambda t_rib, E, NU, RHO:psp.isotropic_plate(thickness=t_rib, E=E, nu=NU, rho=RHO, calc_scf=True)


def eledict(consts:ty.Dict[str,object], sizerVars:ty.Dict[str,object], codes:ty.Dict[str,str]):
    '''a function that initiates them all based on sizer and constant report'''

    eleProps = {"quad":{codes["spar"]:spar_quad(sizerVars["tspar"], consts["E_ALU"], consts["NU"], consts["RHO_ALU"]), 
                        codes["skin"]:spar_quad(sizerVars["tskin"], consts["E_ALU"], consts["NU"], consts["RHO_ALU"]),
                        codes["panelPlate"]:spar_quad(sizerVars["tpan"], consts["E_ALU"], consts["NU"], consts["RHO_ALU"]), 
                        codes["panelRib"]:spar_quad(sizerVars["trib"], consts["E_ALU"], consts["NU"], consts["RHO_ALU"])},
                "spring":{},
                "beam":{codes["rail"]:prop_rail(consts["DIN"], consts["E_STEEL"], consts["RHO_STEEL"]), 
                        codes["panelFlange"]:rib_flange(sizerVars["trib"], consts["RIB_FLANGE"], consts["E_ALU"], consts["RHO_ALU"])}, 
                "mass":{}}
    
    return eleProps
