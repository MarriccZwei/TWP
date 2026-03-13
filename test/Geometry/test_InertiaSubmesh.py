from ...C4.Geometry.InertiaSumesh import InertiaSubmesh
from ...C4.Geometry.ElysianWing import ElysianWing
from ...C4.ConfigFiles.classified import MASSES

import aerosandbox as asb
import aerosandbox.numpy as np
import pyvista as pv

class _SETUP:
    s3 = np.sqrt(3)
    scaffold = np.zeros((3, 6, 3))
    tippts = np.array([[0., 9., 1.], [0., 9., 0.], [2., 9., 2*s3], [3., 9., s3], [5., 9., 3*s3], [5., 9., 0.]])
    fuspts = np.array([[1., 1., 1.], [1., 1., 0.], [5., 1., 4*s3], [7., 1., 2*s3], [8., 1., 3*s3], [8., 1., 0.]])
    scaffold = fuspts[None, :, :]+(tippts-fuspts)[None, :, :]*np.linspace(0., 1., 9)[:, None, None]
    HYPERPARAMS ={
        'delta':.005,
        'D':.95,
        'd':.2,
        'Delta b':.1,
        '(H/c)_sq':.1,
        '(H/c)_aq':.05,
        'rj/c':.1/5
    }

    c_at_y = lambda y:-3/8*y+67/8

    MASSES_ = {
        'rho_bat': 3e3,
        'rho_j':1e3,
        'hi':200.,
        'LE':1e3,
        'TE':928.,
        'bi':17480.,
        'mi':694.625,
        'li':1269.5,
    }

    eqpt_dict = {
        'motor_pts':[(0., 2., .5), (-1, 5., 1.)],
        'motor_is':[1, 4],
        'lg_pt':(8., 3.5, 4.),
        'lg_is':[2, 3]
    }

    ism = InertiaSubmesh(scaffold, HYPERPARAMS, MASSES_, c_at_y, eqpt_dict)


def _plot_inertia_submesh(plotter:pv.Plotter, inesm:InertiaSubmesh):
    for element in inesm.eleNodes:
        if len(element) == 1:
            # Single point
            point = np.array(element[0])
            poly = pv.PolyData(point)
            plotter.add_mesh(poly, color="red", point_size=10, render_points_as_spheres=True)

        elif len(element) == 2:
            # Line between two points
            p1 = np.array(element[0])
            p2 = np.array(element[1])
            line = pv.Line(p1, p2)
            plotter.add_mesh(line, color="blue", line_width=3)

        else:
            raise ValueError(f"Each element must contain either 1 or 2 nodes. Got: {element}")


def test_plot_inertia_submesh():
    plotter = pv.Plotter()
    ism = _SETUP.ism

    _plot_inertia_submesh(plotter, ism)
    print(ism.tot_joint_mass)
        
    plotter.show()


def test_full_geometry():
    HYPERPARAMS ={
        'delta':.005,
        'D':.2,
        'd':.02,
        'Delta b':.1,
        '(H/c)_sq':.0215,
        '(H/c)_aq':.009,
        '(H/c)_pq':.009,
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

    wing = ElysianWing(GEOM_SOURCE, HYPERPARAMS["(H/c)_sq"])
    plotter = pv.Plotter()
    wing.plot(plotter, 9, 20)
    ism = InertiaSubmesh(wing.scaffold, HYPERPARAMS, MASSES, wing.c_at_y, wing.large_equipment_summary())
    print(f"{ism.tot_computed_bat_mass} vs required {MASSES["bi"]}")
    _plot_inertia_submesh(plotter, ism)
    plotter.show()
    print(ism.tot_joint_mass)


if __name__ == "__main__":
    #test_plot_inertia_submesh()
    test_full_geometry()