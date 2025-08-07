import aerosandbox as asb
import typing as ty
import numpy as np
import numpy.typing as nt
from scipy.spatial import cKDTree
import geometricClasses as gcl
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import pypardiso as ppd
from pyfe3d import DOF
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

    return ss.csc_matrix(W), Fext

def fem2aero(les:ty.List[gcl.Point3D], p:nt.NDArray[np.float32], ncoords_s:nt.NDArray[np.float32], ids_s:nt.NDArray[np.int32], N:int, DOF:int,
             number_of_neighbors=2):
    #also skipping the twist for now
    deform_dir = gcl.Direction3D(0,0,1)
    heave_displs =  p[:len(les)]
    leds = [deform_dir.step(le, p_) for le, p_ in zip(les, heave_displs)]

    tree = cKDTree(ncoords_s)
    d, node_indices = tree.query(gcl.pts2numpy3D(leds), k=number_of_neighbors)
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
    if displs is None:
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


def calc_dFv_dp(les:ty.List[gcl.Point3D], tes:ty.List[gcl.Point3D], airfs:ty.List[asb.Airfoil],
        op:asb.OperatingPoint, ymin:float, bres:int=20, cres:int=10, displs:ty.List[float]=None, return_sol=False, epsilon=.01):
    
    airplane, vlm_, forces, moments = vlm(les, tes, airfs, op, bres, cres, displs, return_sol)

    vortex_valid = vlm_.vortex_centers[:, 1] > ymin
    v = vortex_valid.sum()*3
    Fv = vlm_.forces_geometry[vortex_valid].flatten()

    dFv_dp = np.zeros((v, len(displs)*3))
    for i in range(len(airfs)-1):
        p_DOF = 3*(i + 1) + 2 # heave DOF starting at second airfoil
        p2 = displs.copy()
        p2[i+1] += epsilon

        plane2, vlm2, f2, M2 = vlm(les, tes, airfs, op, bres, cres, p2, return_sol)
        Fv2 = vlm2.forces_geometry[vortex_valid].flatten()
        dFv_dp[:, p_DOF] += (Fv2 - Fv)/epsilon
    dFv_dp = ss.csc_matrix(dFv_dp)
    return vlm_, dFv_dp


def flutter_omegans(velocities, M, bu, W_u_to_p, Kuu, W,
                    les:ty.List[gcl.Point3D], tes:ty.List[gcl.Point3D], airfs:ty.List[asb.Airfoil],
        op:asb.OperatingPoint, ymin:float, bres:int=20, cres:int=10):
    omegans=[]
    Muu = M[bu, :][:, bu]
    for velocity in velocities:
        op_pt = asb.OperatingPoint(
        velocity=velocity,      # m/s
        alpha=op.alpha,          # degrees
        beta=op.beta,           # degrees
        atmosphere=op.atmosphere
    )
        #KA *= 1/2*velocity**2*vlm.op_point.atmosphere.density()
        vlm_, dFv_dp = calc_dFv_dp(les, tes, airfs, op_pt, ymin, bres, cres, np.zeros(len(airfs)))
        KA = W @ dFv_dp @ W_u_to_p
        KAuu = KA[bu, :][:, bu]
        k = 2
        tol = 0
        # print(f"KAuu for {velocity} m/s:")
        # print(KAuu)
        # print(f"W_u_to_p for {velocity} m/s:")
        # print(W_u_to_p)
        # print(f"dFv_dp for {velocity} m/s:")
        # print(dFv_dp)
        # print(f"W for {velocity} m/s:")
        # print(W)
        eigvals, peigvecs = ssl.eigs(A=Kuu - KAuu, M=Muu, k=k, which='LM', tol=tol, sigma=-1.)
        omegan = np.sqrt(eigvals)
        omegans.append(omegan)
    return omegans

def flutter_mode(W, W_u_to_p, dFv_dp, Kuu, vlm_, N, ymin, quads, ncoords, nid_pos):
    KA = W @ dFv_dp @ W_u_to_p
    KAuu = KA[bu, :][:, bu]

    vortex_valid = vlm_.vortex_centers[:, 1] > ymin
    Fext = W @ vlm_.forces_geometry[vortex_valid].flatten()

    u = np.zeros(N)
    uu = ppd.spsolve(Kuu - KAuu, Fext[bu])
    u[bu] = uu

    scale = 1e6
    displ_vec = np.zeros((N, 3))
    for i in range(3):
        displ_vec[:, i] = scale*u[i::DOF]
    tot_displ =  np.sqrt(displ_vec[:, 0]**2 + displ_vec[:, 1]**2 + displ_vec[:, 2]**2)

    plot3d_pyvista(ncoords, nid_pos, quads, displ_vec, tot_displ)

def plot3d_pyvista(ncoords, nid_pos, quads, displ_vec=None, contour_vec=None,
        contour_label='vec', contour_colorscale='coolwarm', background='white'):
    """Plot results using pyvista
 
    Parameters
    ----------
    displ_vec : array-like
        Nodal displacements to be applied to the nodal coordinates in order to visualize the deformed model.
    contour_vec : array-like
        Nodal or element-wise output to be displayed as a contour plot on top of the undeformed mesh.
        For nodal output, the function expects the order of attr:`.ModelPyFE3D.ncoords`.
        For element-wise output, the function expects the element outputs in the following order of elements:
 
        - :attr:`ModelPyFE3D.list_Quad4R`
        - :attr:`ModelPyFE3D.list_Tria3R`
 
    contour_label : str
        Name of contour being displayed.
    contour_colorscale : str
        Name of the colorscale.
 
    Returns
    -------
    plotter : `pyvista.Plotter` object
        A plotter object that is ready to be modified, or shown with ``plotter.show()``.
 
    """
    import pyvista as pv
 
    # NOTE needs ipygany and pythreejs packages (pip install pyvista pythreejs ipygany --upgrade)
 
    intensitymode = None
    if contour_vec is not None:
        if len(contour_vec) == len(ncoords):
            intensitymode = 'vertex'
        elif len(contour_vec) == len(quads):
            intensitymode = 'cell'
        else:
            raise RuntimeError('coutour_vec must be either a nodal or element output')
 
    plotter = pv.Plotter(off_screen=False)
    faces_quad = []
    for q in quads:
        faces_quad.append([4, nid_pos[q.n1], nid_pos[q.n2], nid_pos[q.n3], nid_pos[q.n4]])
    faces_quad = np.array(faces_quad)
    quad_plot = pv.PolyData(ncoords, faces_quad)
    if contour_vec is not None:
        if intensitymode == 'vertex':
            quad_plot[contour_label] = contour_vec
        else:
            quad_plot[contour_label] = contour_vec[:len(quads)]
 
        plotter.add_mesh(quad_plot, scalars=contour_label,
                cmap=contour_colorscale, edge_color='black', show_edges=True,
                line_width=1.)
    else:
        plotter.add_mesh(quad_plot, edge_color='black', show_edges=True,
                line_width=1.)
        
    if displ_vec is not None:
        quad_plot = pv.PolyData(ncoords + displ_vec, faces_quad)
        plotter.add_mesh(quad_plot, edge_color='red', show_edges=True,
                line_width=1., opacity=0.5)
        
    plotter.set_background(background)
    plotter.parallel_projection = False
    return plotter


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
    check = (x>0.8) & (x<1.5)
    ncoords_s = ncoords[check, :]
    ids_s = np.array(range(ncoords.shape[0]), dtype=np.int32)[check]
    bk = np.zeros(6*ncoords.shape[0], dtype=np.bool_)

    checky = y>ncoords[:, 1].min()
    for i in range(6):
        bk[i::6] = checky
    bu = ~bk

    #aerodynamic model
    op = asb.OperatingPoint(asb.Atmosphere(altitude=7000), 100, 5, 0)
    airplane, vlm_, forces, moments = vlm(les, tes, [asb.Airfoil("naca2412")]*6, op)
    W, fext = aero2fem(vlm_, ncoords_s, ids_s, 6*ncoords.shape[0], 6)
    fz = fext[2::6]

    #plotting
    import pyvista as pv
    import matplotlib.pyplot as plt

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

    vlm_, dFv_dp = calc_dFv_dp(les, tes, [asb.Airfoil("naca2412")]*6, op, ncoords_s[:, 1].min(), displs=np.zeros(6))
    print(dFv_dp)
    W_u_to_p = fem2aero(les, np.zeros(6), ncoords_s, ids_s, 6*ncoords.shape[0], 6)
    velocities = np.linspace(30, 200, 10)
    omegans = flutter_omegans(velocities, np.eye(6*ncoords.shape[0]), bu, W_u_to_p, 
                              np.eye(6*ncoords.shape[0])[bu, :][:, bu], W, les, tes, [asb.Airfoil("naca2412")]*6,
                                op, ncoords_s[:, 1].min())
    plt.figure(figsize=(5,5))
    plt.plot(velocities, omegans)
    plt.show()
