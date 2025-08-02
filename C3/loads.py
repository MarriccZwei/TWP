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


if __name__ == "__main__"