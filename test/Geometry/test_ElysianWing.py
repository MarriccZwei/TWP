from ...C4.Geometry.ElysianWing import ElysianWing

import pyvista as pv
import aerosandbox as asb

def test_ElysianWing():
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

    H_per_c_sq = .01
    wing = ElysianWing(GEOM_SOURCE, H_per_c_sq, debug=True)
    plotter = pv.Plotter()
    wing.plot(plotter)
    print(wing.aero_foils(14, plotter))
    plotter.show()

if __name__ == "__main__":
    test_ElysianWing()