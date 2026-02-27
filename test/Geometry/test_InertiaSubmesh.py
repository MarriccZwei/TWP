from ...C4.Geometry.InertiaSumesh import InertiaSubmesh

import aerosandbox.numpy as np
import pyvista as pv

def test_battery_packing():
    s3 = np.sqrt(3)
    scaffold = np.zeros((3, 6, 3))
    tippts = np.array([[0., 9., 1.], [0., 9., 0.], [2., 9., 2*s3], [3., 9., s3], [5., 9., 3*s3], [5., 9., 0.]])
    fuspts = np.array([[1., 1., 1.], [1., 1., 0.], [5., 1., 4*s3], [7., 1., 2*s3], [8., 1., 3*s3], [8., 1., 0.]])
    scaffold = fuspts[None, :, :]+(tippts-fuspts)[None, :, :]*np.linspace(0., 1., 9)[:, None, None]

    plotter = pv.Plotter()
    scaffold_points = scaffold.reshape((scaffold.shape[0]*scaffold.shape[1], scaffold.shape[2]))
    plotter.add_points(scaffold_points, color="red", point_size=8, render_points_as_spheres = True)

    HYPERPARAMS ={
        'delta':.005,
        'D':.150,
        'd':.2,
        'Delta b':.1,
        '(H/c)_sk':.1,
        '(H/c)_as':.005,
    }

    c_at_y = lambda y:-3/8*y+67/8

    MASSES = {}

    ism = InertiaSubmesh(scaffold, HYPERPARAMS, MASSES, c_at_y)
    ism.plot_bats(plotter)

    plotter.add_axes(
    line_width=3,
    labels_off=False
    )
    plotter.show_grid()
    plotter.show()

if __name__ == "__main__":
    test_battery_packing()