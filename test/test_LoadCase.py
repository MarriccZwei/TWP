from ..C4.LoadCase import LoadCase 
from ..C4.Geometry.geometryInit import geometry_init
from ..C4.ConfigFiles.classified import MASSES

import aerosandbox as asb
import aerosandbox.numpy as np

def test_compressibility_corrections():
    for i in range(8, 12):
        lc_dummy = LoadCase(1., 100., 36, 9.81, 0., asb.OperatingPoint(velocity=270.), np.array([[0., 0., 0.], [0., 1., 0.], [0., 2., 0.]]), 
                            np.array([[0.2, 0., 0.], [0.2, 1., 0.], [0.2, 2., 0.]]), [asb.Airfoil("naca5412")]*3, 20, i, 10, True)
        airplane, vlm, forces, moments = lc_dummy._vlm(debug=True)
        print(forces)
    vlm.draw()

def test_moments():
    lc_dummy = LoadCase(1., 100., 36, 9.81, 0., asb.OperatingPoint(velocity=270.), np.array([[0., 0., 0.], [0., 1., 0.], [0., 2., 0.]]), 
                            np.array([[0.2, 0., 0.], [0.2, 1., 0.], [0.2, 2., 0.]]), [asb.Airfoil("naca5412")]*3, 20, 10, 10, True)
    
    HYPERPARAMS ={
        'delta':.005,
        'D':.3,
        'd':.02,
        'Delta b':.1,
        '(H/c)_sq':.009,
        '(H/c)_aq':.003,
        '(H/c)_pq':.006,
        'rj/c':.1/5
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

    N = 7
    model, mesher, excl, wing = geometry_init(GEOM_SOURCE, HYPERPARAMS, MASSES, N, 8)

    

if __name__ == "__main__":
    test_compressibility_corrections()