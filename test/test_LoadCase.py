from ..C4.LoadCase import LoadCase 
from ..C4.Geometry.geometryInit import geometry_init
from ..C4.ConfigFiles.classified import MASSES

import aerosandbox as asb
import aerosandbox.numpy as np
import pyfe3d as pf3
import matplotlib.pyplot as plt
import pyvista as pv

def test_compressibility_corrections():
    for i in range(8, 12):
        lc_dummy = LoadCase(1., 100., 36, 9.81, 0., asb.OperatingPoint(velocity=270.), np.array([[0., 0., 0.], [0., 1., 0.], [0., 2., 0.]]), 
                            np.array([[0.2, 0., 0.], [0.2, 1., 0.], [0.2, 2., 0.]]), [asb.Airfoil("naca5412")]*3, 20, i, 10, True)
        airplane, vlm, forces, moments = lc_dummy._vlm(debug=True)
        print(forces)
    vlm.draw()


def test_moments(plot=False):
    #see if the pitching moment stays equivalent after the interpolation
    
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
    airfs, les, tes = wing.aero_foils(10)

    lc_dummy = LoadCase(1., 76000., model.N, 9.81, 0., asb.OperatingPoint(velocity=269., alpha=-.75, atmosphere=asb.Atmosphere(7000)), les, tes, airfs, 20, 10, 10, True)

    #total moment as applied to the structure
    lc_dummy.apply_aero(*mesher.get_submesh('sq'))
    moments_applied = lc_dummy.A[2::pf3.DOF]*model.x
    mom_appl = moments_applied.sum()
    
    airplane, vlm, forces, moments = lc_dummy._vlm(debug=True)
    vortex_valid = vlm.vortex_centers[:, 1] > model.y.min()
    Fvz = vlm.forces_geometry[vortex_valid][:, 2]
    moments_vlm = Fvz*vlm.vortex_centers[vortex_valid][:, 0]
    mom_vlm = moments_vlm.sum()

    if plot:
        plt.scatter(model.x, model.y, label="model nodes")
        plt.scatter(vlm.vortex_centers[vortex_valid][:, 0], vlm.vortex_centers[vortex_valid][:, 1], label="vortices")
        plt.legend()
        plt.xlabel('x from LE [m]')
        plt.ylabel('y [m]')
        plt.show()

    assert np.isclose(mom_appl, mom_vlm, rtol=1e-1), f"applied: {mom_appl}; from vlm: {mom_vlm}"
    

def plot_upward_arrows(
    points,
    magnitudes,
    factor=1.0,
    normalize=False,
    arrow_kwargs=None,
    show_points=True,
    color=None,
    scalars=None,
    cmap="viridis"
):
    """
    Plot upward arrows at given 3D points with lengths based on magnitudes.

    Parameters
    ----------
    points : (n, 3) array-like
    magnitudes : (n,) array-like

    factor : float
        Global scaling factor.

    normalize : bool
        Normalize magnitudes to [0, 1].

    arrow_kwargs : dict
        Passed to pv.Arrow() (geometry only).

    show_points : bool
        Show base points.

    color : str or tuple, optional
        Solid color for all arrows (e.g., "red" or (1,0,0)).

    scalars : (n,) array-like, optional
        Values used to color arrows individually.

    cmap : str
        Colormap name if scalars are provided.
    """

    points = np.asarray(points)
    magnitudes = np.asarray(magnitudes)

    if normalize:
        max_val = np.max(magnitudes)
        if max_val > 0:
            magnitudes = magnitudes / max_val

    mesh = pv.PolyData(points)

    directions = np.tile([0, 0, 1], (points.shape[0], 1))

    mesh["vectors"] = directions
    mesh["magnitudes"] = magnitudes

    # Attach scalars for coloring if provided
    if scalars is not None:
        mesh["colors"] = np.asarray(scalars)
        color_array_name = "colors"
    else:
        color_array_name = None

    arrow_kwargs = arrow_kwargs or {}
    arrow_geom = pv.Arrow(**arrow_kwargs)

    arrows = mesh.glyph(
        orient="vectors",
        scale="magnitudes",
        factor=factor,
        geom=arrow_geom
    )

    plotter = pv.Plotter()

    if scalars is not None:
        plotter.add_mesh(arrows, scalars=color_array_name, cmap=cmap)
    else:
        plotter.add_mesh(arrows, color=color or "orange")

    if show_points:
        plotter.add_points(points, color="blue", point_size=8)

    plotter.show()

    return plotter


if __name__ == "__main__":
    #test_compressibility_corrections()
    test_moments(True)