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
        "deltaxlg":.4039,
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

    N = 7

    LC_INFO = [
    {
        'n':2.5,
        'nlg':0.,
        'Ttot':112800., # [N]
        'op':asb.OperatingPoint(asb.Atmosphere(0.), velocity=90., alpha=10.), #[h]=m, [v]=m/s, [alpha]=deg
        'aeroelastic':False
    },
    {
        'n':-1.,
        'nlg':0.,
        'Ttot':32400., # [N]
        'op':asb.OperatingPoint(asb.Atmosphere(7000.), velocity=187., alpha=-4.5), #[h]=m, [v]=m/s, [alpha]=deg
        'aeroelastic':False
    },
    {
        'n':1.,
        'nlg':1.5,
        'Ttot':37800., # [N]
        'op':asb.OperatingPoint(asb.Atmosphere(7000.), velocity=269., alpha=-.75), #[h]=m, [v]=m/s, [alpha]=deg
        'aeroelastic':True
    },
    ]

    G0 = 9.81 # [N/kg]
    MTOM = 76000. # [kg]

    model, mesher, excl, wing, ism = geometry_init(GEOM_SOURCE, HYPERPARAMS, MASSES, N, LC_INFO, G0, MTOM, 8)

    desvars = {
        '(2t/H)_Sq':0.3,
        '(2t/H)_Pq':0.25,
        '(2t/H)_Aq':0.1,
        'W_bb':0.01,
        'W_mb':0.01,
        'W_lb':0.02,
        'ds':.01,
        'de':.01,
        '(2t/H)_sq':0.06,
        '(2t/H)_pq':0.03,
        '(2t/H)_aq':0.03
    }

    materials = {
        'E_ALU':72.4e9,
        'NU_ALU':.33,
        'RHO_ALU':2780.,
        'SF_ALU':323.33e6,
        'E_FOAM':0.58644e9,
        'NU_FOAM':.275,
        'RHO_FOAM':250.2,
        'SF_FOAM':3.645e6
    }

    ep = load_ele_props(desvars, materials, mesher.eleTypes, mesher.eleArgs)

    beamprops = ep["beamprops"]
    beamorients = ep["beamorients"]
    shellprops = ep["shellprops"]
    matdirs = ep["matdirs"]
    inertia_vals = ep["inertia_vals"]

    model.load_props(beamprops, beamorients, shellprops, matdirs, inertia_vals)
    # model.KC0_M_update(beamprops, beamorients, shellprops, matdirs, [0]*len(inertia_vals)) #optional removing bending relief

    airfs, les, tes = wing.aero_foils(10)

    lc = LoadCase(1., 76000, model.N, 9.81, 112800, asb.OperatingPoint(asb.Atmosphere(0.), alpha=15., velocity=74.), les, tes, airfs, 
                  nneighs=10, cres=10, bres=20, nlg=1.5, bank=6.)
    # lc = LoadCase(-1., 76000, model.N, 9.81, 32400., asb.OperatingPoint(asb.Atmosphere(7000.), alpha=-4.5, velocity=187.), les, tes, airfs, 
    #               nneighs=5, cres=8, bres=20, nlg=0., bank=0.)
    lc.apply_aero(*mesher.get_submesh_list(['sq', 'Sq']), debug=True)
    lc.apply_thrust(mesher.get_submesh('mi')[0])
    lc.apply_landing(mesher.get_submesh('li')[0])
    ctrl_sum_L = np.sqrt(lc.L[0::pf3.DOF].sum()**2+lc.L[1::pf3.DOF].sum()**2+lc.L[2::pf3.DOF].sum()**2)
    ctrl_sum_A_z = lc.n*G0*MTOM/2
    assert np.isclose(ctrl_sum_L, 1118340./2), ctrl_sum_L
    print(f"applied: {lc.A[2::pf3.DOF].sum()} < (slightly) nominal: {ctrl_sum_A_z}")
    
    #post processing
    savePath = FW_SAVE_PATH
    print("processing starts")
    margins = process_load_case(model, lc, materials, desvars, ep["beamtypes"], ep["quadtypes"], excl, plot=True, savePath=savePath, num_eig_lb=0)
    print(margins)
    print(model.ncoords.shape) #so that it can be compared with the shape from CATIA


if __name__ == "__main__":
    test_self_weight()
    