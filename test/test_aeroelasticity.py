from ..C4.Geometry.geometryInit import geometry_init
from ..C4.Solution.eleProps import load_ele_props
from ..C4.Solution.processLoadCase import process_aeroelastic_load_case
from ..C4.LoadCase import LoadCase
from ..C4.ConfigFiles.classified import MASSES

import aerosandbox as asb 
import pyfe3d as pf3
import numpy as np
import aerosandbox as asb

def test_self_weight():
    HYPERPARAMS ={
        'delta':.005,
        'D':.3,
        'd':.02,
        'Delta b':.1,
        '(H/c)_sq':.009,
        '(H/c)_aq':.003,
        '(H/c)_pq':.006
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

    model, mesher, excl = geometry_init(GEOM_SOURCE, HYPERPARAMS, MASSES, N, 8)

    desvars = {
        '(2t/H)_sq':0.1,
        '(2t/H)_pq':0.05,
        '(2t/H)_aq':0.05,
        'W_bb':0.003,
        'W_mb':0.003,
        'W_lb':0.003,
        'Ds':.008
    }

    materials = {
        'E_ALU':72.4e9,
        'NU_ALU':.33,
        'RHO_ALU':2780.,
        'SF_ALU':323.33e6,
        'E_FOAM':4.93e9,
        'NU_FOAM':.214,
        'RHO_FOAM':433.,
        'SF_FOAM':31.5e6
    }

    ep = load_ele_props(desvars, materials, mesher.eleTypes, mesher.eleArgs)

    beamprops = ep["beamprops"]
    beamorients = ep["beamorients"]
    shellprops = ep["shellprops"]
    matdirs = ep["matdirs"]
    inertia_vals = ep["inertia_vals"]

    model.KC0_M_update(beamprops, beamorients, shellprops, matdirs, inertia_vals)

    les_str = "0.0, 0.0, 0.0,0.24796410315737183, 2.6250000000000013, 0.21803733205038336,0.49592820631474366, 5.250000000000003, 0.43607466410076673,0.7438923094721154, 7.875000000000002, 0.6541119961511499,0.9918564126294873, 10.500000000000005, 0.8721493282015335,1.2398205157868587, 13.125000000000002, 1.0901866602519166,1.4877846189442308, 15.750000000000004, 1.3082239923022998,1.7357487221016028, 18.37500000000001, 1.5262613243526835,1.9837128252589746, 21.00000000000001, 1.744298656403067"
    tes_str = "4.950622318602955, 0.0, 0.009334779044329352,4.827275918562817, 2.6250000000000013, 0.22646456313206953,4.7039295185226795, 5.250000000000003, 0.4435943472198098,4.580583118482542, 7.875000000000004, 0.66072413130755,4.457236718442404, 10.500000000000005, 0.8778539153952902,4.333890318402267, 13.125000000000005, 1.09498369948303,4.210543918362128, 15.750000000000009, 1.3121134835707706,4.087197518321991, 18.37500000000001, 1.5292432676585108,3.963851118281853, 21.00000000000001, 1.7463730517462508"
    les_flat = np.fromstring(les_str, sep=",")
    tes_flat = np.fromstring(tes_str, sep=",")
    les = les_flat.reshape((len(les_flat)//3, 3))
    tes = tes_flat.reshape((len(les_flat)//3, 3)) #should have same length
    airfs = [asb.Airfoil(f"naca241{i}") for i in reversed(range(9))] #from naca 2418 to naca 2410

    lc = LoadCase(1., 76000, model.N, 9.81, 112800, asb.OperatingPoint(atmosphere=asb.Atmosphere(7000), alpha=.87, velocity=269.), les, tes, airfs, bres=30, cres=8, aeroelastic=True, nneighs=10)
    #lc.apply_aero(*mesher.get_submesh('sq'))
    lc.aerodynamic_matrix(*mesher.get_submesh('sq'))
    print(model.KC0[model.KC0>0.].mean(), np.abs(lc.KA[np.abs(lc.KA)>0.]).mean())

    #post processing
    savePath = r"C:\marek\studia\hpb\Results\C4\ForwardTests\\"
    omegan = process_aeroelastic_load_case(model, lc, plot=True, savePath=savePath, k=30, returnOmegan=True)
    print(omegan)


if __name__ == "__main__":
    test_self_weight()
    