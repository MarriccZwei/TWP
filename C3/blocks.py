import ptsFromCAD as pfc
import geometricClasses as gcl
import meshingComponents as mc

import typing as ty

def mesh_block(cadData:str, sizerVars:ty.Dict[str:str], eleProps:ty.Dict[str,ty.Dict[str, object]], consts:ty.Dict[str, int]):
    up = pfc.UnpackedPoints(cadData)
    mesh = gcl.Mesh3D()
    pts, ids = mc.all_components(mesh, up, consts["NB_COEFF"], consts["NA"], consts["NF2"], consts["NIP_COEFF"], consts["NTRIG"], 
                                 dz, din, cspacing, bspacing, BAT_MASS_1WING, lemass, temass,
                             spar, panelPlate, panelRib, panelFlange, skin, batteryRail, battery, motor, lg, hinge, mount)
