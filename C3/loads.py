import aerosandbox as asb
import typing as ty
import numpy as np
import numpy.typing as nt
from scipy.spatial import cKDTree
import geometricClasses as gcl
import scipy.sparse as ss
"""This module shall contain functions that take the force vector f, ncoords, x, y, z 
plus additional data and apply aerodynamic, propulsive, landing an inertial loads."""

def aero2fem(vlm:asb.VortexLatticeMethod, ncoords_s:nt.NDArray[np.float32], ids_s:nt.NDArray[np.int32], N:int, DOF:int,
             number_of_neighbors=10):
    """ncoords selected - only selected nodes to which we apply the loads - will have to be tracked externally with ids_s"""

    # x, y, z coordinates of each node, NOTE assuming same order for DOFs
    vortex_valid = vlm.vortex_centers[:, 1] > ncoords_s[:, 1].min() # region represented by the FE model

    tree = cKDTree(ncoords_s)
    d, node_indices = tree.query(vlm.vortex_centers[vortex_valid], k=number_of_neighbors)

    power = 2
    weights = (1/d**power)/((1/d**power).sum(axis=1)[:, None])
    assert np.allclose(weights.sum(axis=1), 1)

    v = vortex_valid.sum()*3
    W = np.zeros((N, v))

    W *= 0
    for j in range(number_of_neighbors):
        for i, node_index in enumerate(node_indices[:, j]):
            #Here we cannot use the node_index directly, as we need to match the node indexing in the global mesh
            W[DOF*ids_s[node_index] + 0, i*3 + 0] += weights[i, j]
            W[DOF*ids_s[node_index] + 1, i*3 + 1] += weights[i, j]
            W[DOF*ids_s[node_index] + 2, i*3 + 2] += weights[i, j]
    assert np.allclose(W.sum(axis=0), 1.)

    Fv = vlm.forces_geometry[vortex_valid].flatten()
    Fext = W @ Fv #external forces to be applied to the fem model

    assert np.isclose(Fext[0::DOF].sum(), Fv[0::3].sum())
    assert np.isclose(Fext[1::DOF].sum(), Fv[1::3].sum())
    assert np.isclose(Fext[2::DOF].sum(), Fv[2::3].sum())

    return W, Fext

def fem2aero(les:ty.List[gcl.Point3D], p:nt.NDArray[np.float32], ncoords_s:nt.NDArray[np.float32], ids_s:nt.NDArray[np.int32], N:int, DOF:int,
             number_of_neighbors=2):
    #also skipping the twist for now
    deform_dir = gcl.Direction3D(0,0,1)
    heave_displs =  p[:len(les)]
    leds = [deform_dir.step(le, p_) for le, p_ in zip(les, heave_displs)]

    tree = cKDTree(ncoords_s)
    d, node_indices = tree.query(leds, k=number_of_neighbors)
    power = 2
    weights2 = (1/d**power)/((1/d**power).sum(axis=1)[:, None])
    assert np.allclose(weights2.sum(axis=1), 1)

    W_u_to_p = np.zeros((len(heave_displs)*3, N))

    for j in range(number_of_neighbors):
        for i, node_index in enumerate(node_indices):
            W_u_to_p[i*3 + 0, DOF*node_index + 0] = weights2[i, j]
            W_u_to_p[i*3 + 1, DOF*node_index + 1] = weights2[i, j]
            W_u_to_p[i*3 + 2, DOF*node_index + 2] = weights2[i, j]
        
    return ss.csc_matrix(W_u_to_p)

def vlm(les:ty.List[gcl.Point3D], tes:ty.List[gcl.Point3D], airfs:ty.List[asb.Airfoil],
        op:asb.OperatingPoint, bres:int=20, cres:int=10, displs:ty.List[float]=None, return_sol=False):
    #allowing for initial displacements for flutter. Yet, for static analysis we dont's need displacements
    if displs==None:
        displs = np.zeros(len(airfs))
    heaveDir = gcl.Direction3D(0,0,1)
    cs = [te.x-le.x for te, le in zip(tes, les)] #obtaining the chord list
    ptles = [heaveDir.step(le, displ) for le, displ in zip(les, displs)]

    airplane = asb.Airplane("E9X", xyz_ref=[0,0,0],
                            wings = [
                                asb.Wing(
                                    name="E9X_WING",
                                    symmetric=True,
                                    xsecs=[asb.WingXSec(
                                        xyz_le=[ptle.x, ptle.y, ptle.z],
                                        chord=c,
                                        airfoil=airf
                                    )for ptle, c, airf, in zip(ptles, cs, airfs)],
                                )
                            ])
    vlm = asb.VortexLatticeMethod(airplane, op, spanwise_resolution=bres, chordwise_resolution=cres)
    sol = vlm.run()
    forces = sol["F_g"]
    moments = sol["M_g"]

    if return_sol:
        return airplane, vlm, forces, moments, sol
    else:
        return airplane, vlm, forces, moments


if __name__ == "__main__":
    #geometry definition
    les = [gcl.Point3D(x_, y_, z_) for x_, y_, z_ in zip(
        [0, 0.1, 0.2, 0.5, 0.6, 0.7], [0, 1, 2, 6, 9, 10], [0, 0.009, 0.01, 0.03, 0.05, 0.075]
    )]
    cs = [2.3, 2, 1.8, 1, 0.8]
    dirc = gcl.Direction3D(1,0,0)
    tes = [dirc.step(le, c) for le, c in zip(les, cs)]

    #structural model - not spanning the entire wing
    sheet = gcl.multi_section_sheet3D(les[1:5], tes[1:5], 13, [5,10,5])
    ncoords = np.array([[pt.x, pt.y, pt.z] for pt in sheet.flatten()]) #artifficial ncoords
    x, y, z = ncoords[:,0], ncoords[:,1], ncoords[:, 2]
    check = (x>0.5) & (x<1.5)
    ncoords_s = ncoords[check, :]
    ids_s = np.array(range(ncoords.shape[0]), dtype=np.int32)[check]

    #aerodynamic model
    op = asb.OperatingPoint(asb.Atmosphere(altitude=7000), 100, 5, 0)
    airplane, vlm_, forces, moments = vlm(les, tes, [asb.Airfoil("naca2412")]*6, op)
    W, fext = aero2fem(vlm_, ncoords_s, ids_s, 6*ncoords.shape[0], 6)
    fz = fext[2::6]

    #plotting
    import pyvista as pv

    airplane.draw()

    # Create a PyVista point cloud (PolyData)
    point_cloud = pv.PolyData(ncoords)

    # Plot it
    point_cloud.plot(render_points_as_spheres=True, point_size=5)

    # Create a uniform direction array (positive Z direction)
    directions = np.tile([0, 0, 1], (ncoords.shape[0], 1)) * fz[:, np.newaxis]/fz.max()

    point_cloud = pv.PolyData(ncoords)
    # Set the direction vectors as a field
    point_cloud["vectors"] = directions

    # Create glyphs (arrows) from vectors
    arrows = point_cloud.glyph(orient="vectors", scale="vectors", factor=1.0)

    # Plot the arrows
    arrows.plot()