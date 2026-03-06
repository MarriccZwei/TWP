import aerosandbox as asb
import aerosandbox.numpy as np
import pyvista as pv

from ...C4.Geometry.ElysianWing import ElysianWing
from ...C4.Geometry.StructuralSubmesh import StructuralSubmesh

class _SETUP:
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

    N = 6

    wing = ElysianWing(GEOM_SOURCE, HYPERPARAMS["(H/c)_sq"])
    ssm = StructuralSubmesh(wing, HYPERPARAMS, N)


def test_ssm():
    plotter = pv.Plotter()

    # Storage for global geometry
    line_points = []
    line_cells = []

    quad_points = []
    quad_cells = []

    line_point_offset = 0
    quad_point_offset = 0

    for element in _SETUP.ssm.eleNodes:

        if len(element) == 2:
            # ----- Line -----
            pts = np.array(element)
            line_points.extend(pts)

            # VTK line format: [2, id0, id1]
            line_cells.extend([2,
                               line_point_offset,
                               line_point_offset + 1])

            line_point_offset += 2

        elif len(element) == 4:
            # ----- Quad -----
            pts = np.array(element)
            quad_points.extend(pts)

            # VTK quad format: [4, id0, id1, id2, id3]
            quad_cells.extend([4,
                               quad_point_offset,
                               quad_point_offset + 1,
                               quad_point_offset + 2,
                               quad_point_offset + 3])

            quad_point_offset += 4

        else:
            raise ValueError("Elements must have length 2 (line) or 4 (quad).")

    # ----- Create and add line mesh -----
    if line_points:
        line_points = np.array(line_points)
        line_cells = np.array(line_cells)
        lines = pv.PolyData()
        lines.points = line_points
        lines.lines = line_cells
        plotter.add_mesh(lines, color="blue", line_width=2)

    # ----- Create and add quad mesh -----
    if quad_points:
        quad_points = np.array(quad_points)
        quad_cells = np.array(quad_cells)
        quads = pv.PolyData()
        quads.points = quad_points
        quads.faces = quad_cells
        plotter.add_mesh(
            quads,
            color="brown",
            show_edges=True,
            edge_color="black",
            opacity=0.8,
        )

    plotter.show()

if __name__ == "__main__":
    test_ssm()