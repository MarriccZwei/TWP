from ..C4.Geometry.geometryInit import geometry_init
from ..C4.Solution.eleProps import load_ele_props
from ..C4.Solution.processLoadCase import process_load_case
from ..C4.LoadCase import LoadCase
from ..C4.ConfigFiles.classified import MASSES
from ..C4.ConfigFiles.userConfig import FW_SAVE_PATH

import aerosandbox as asb 
import pyfe3d as pf3
#import scipy.sparse.linalg as ssl
from pypardiso import spsolve
import numpy as np
import pyvista as pv
import aerosandbox as asb

def test_self_weight():
    HYPERPARAMS ={
        'delta':.005,
        'D':.3,
        'd':.03,
        'Delta b':.1,
        '(H/c)_sq':.009,
        '(H/c)_aq':.003,
        '(H/c)_pq':.006,
        'rj/c':.1/5,
    }

    GEOM_SOURCE ={
        #NOTE: all coordinates in m
        "yfus":1.602374,
        "yhn":18.,
        "ytip":21.,
        "deltazhn":1.362017,
        "deltaxhn":1.548961,
        "(x/c)_fore":.15,
        "(x/c)_rear":.7,
        "cr":5.,
        "ct":2.,
        "ylg":5.768546,
        "deltaxlg":.80115,
        "rlg":.801187,
        "ym1":4.005935,
        "ym2":8.091988,
        "ym3":12.178042,
        "ym4":16.290801,
        "deltaxm1":-.801187,
        "deltaxm2":-.267062,
        "deltaxm3":.267062,
        "deltaxm4":.801187,
        "rootfoil":asb.Airfoil("naca2418"),
        "tipfoil":asb.Airfoil("naca2410")
    }

    N = 5

    model, mesher, excl, wing = geometry_init(GEOM_SOURCE, HYPERPARAMS, MASSES, N, 8)

    desvars = {
        '(2t/H)_sq':0.07,
        '(2t/H)_pq':0.01,
        '(2t/H)_aq':0.01,
        'W_bb':0.004,
        'W_mb':0.004,
        'W_lb':0.004,
        'Ds':.008,
    }

    materials = {
        'E_ALU':72.4e9,
        'NU_ALU':.33,
        'RHO_ALU':2780.,
        'SF_ALU':323.33e6,
        'E_FOAM':6.68e9,
        'NU_FOAM':.275,
        'RHO_FOAM':688.,
        'SF_FOAM':45e6
    }

    ep = load_ele_props(desvars, materials, mesher.eleTypes, mesher.eleArgs)

    beamprops = ep["beamprops"]
    beamorients = ep["beamorients"]
    shellprops = ep["shellprops"]
    matdirs = ep["matdirs"]
    inertia_vals = ep["inertia_vals"]

    model.KC0_M_update(beamprops, beamorients, shellprops, matdirs, inertia_vals)

    airfs, les, tes = wing.aero_foils(10)

    lc = LoadCase(2.5, 76000, model.N, 9.81, 112800, asb.OperatingPoint(asb.Atmosphere(0.), alpha=10., velocity=90.), les, tes, airfs, nneighs=5, cres=8, bres=20)
    lc.apply_aero(*mesher.get_submesh('sq'))
    lc.apply_thrust(mesher.get_submesh('mi')[0])
    
    #post processing
    savePath = FW_SAVE_PATH
    print("processing starts")
    margins = process_load_case(model, lc, materials, desvars, ep["beamtypes"], ep["quadtypes"], excl, plot=True, savePath=savePath, num_eig_lb=0)
    print(margins)
    print(model.ncoords.shape) #so that it can be compared with the shape from CATIA


if __name__ == "__main__":
    test_self_weight()
    