from ...C4.Geometry.geometryInit import geometry_init
import pyvista as pv
import numpy as np
import aerosandbox as asb

def test_geometry_init():
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

    MASSES = {
        'rho_bat': 3e3,
        'hi':200.,
        'LE':1e3,
        'TE':928.,
        'bi':17480.,
        'mi':694.625,
        'li':1269.5,
    }

    N = 5

    model, mesher = geometry_init(GEOM_SOURCE, HYPERPARAMS, MASSES, N)

    #quads
    cells = list()
    for quad in model.quads:
        #print(quad.n3)
        cells.append([4, model.nid_pos[quad.n1], model.nid_pos[quad.n2], model.nid_pos[quad.n3], model.nid_pos[quad.n4]])
    cells = np.array(cells).flatten()

    cell_types = np.full(len(model.quads), pv.CellType.QUAD)
    mesh = pv.UnstructuredGrid(cells, cell_types, model.ncoords)

    #beams
    cellsb = list()
    for beam in model.beams:
        cellsb.append([2, model.nid_pos[beam.n1], model.nid_pos[beam.n2]])
    cellsb = np.array(cellsb)

    cellb_types = np.full(len(model.beams), pv.CellType.LINE)
    meshb = pv.UnstructuredGrid(cellsb, cellb_types, model.ncoords)

    #springs
    cellss = list()
    for spring in model.springs:
        cellss.append([2, model.nid_pos[spring.n1], model.nid_pos[spring.n2]])
    cellss = np.array(cellss)
    cells_types = np.full(len(model.springs), pv.CellType.LINE)
    meshs = pv.UnstructuredGrid(cellss, cells_types, model.ncoords)

    pts = list()
    for ine in model.inertia_poses:
        pts.append(model.ncoords[ine,:])
    pts = np.array(pts)
    cloud = pv.PolyData(pts)

    plotter = pv.Plotter()
    
    plotter.add_mesh(
        mesh,
        show_edges=True,
        color="lightblue",
        edge_color="black"
    )

    plotter.add_mesh(
        meshb,
        show_edges=True,
        color="red",
        edge_color="black"
    )

    plotter.add_mesh(
        meshs,
        show_edges=True,
        color="yellow",
        edge_color="black"
    )

    plotter.add_mesh(
        cloud,
        point_size=6,
        render_points_as_spheres=True,
        cmap="viridis"
    )

    plotter.show()

if __name__ == "__main__":
    test_geometry_init()