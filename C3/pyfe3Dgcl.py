import numpy as np
import pyfe3d.shellprop_utils as psu
import pyfe3d as pf3
import scipy.sparse as ss
import typing as ty
import geometricClasses as gcl

"""The file containing an interpreter of the gcl connection matrix into pyfe3D elements"""
def eles_from_gcl(mesh:gcl.Mesh3D, eleDict:ty.Dict[str, ty.Dict[str, object]]):
    #currently only for the elements we have used, same goes to gcl connections
    probes = {"quad": pf3.Quad4Probe, "spring": pf3.SpringProbe, "mass":None}
    data = {"quad": pf3.Quad4Data, "spring": pf3.SpringData, "mass":None}

    #obtaining nodes geometry
    ncoords, x, y, z, ncoords_flatten = mesh.pyfe3D()

    #converting to pyfe3D id system
    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))
    num_elements = len(mesh.connections)

    #initialising system matrices
    #data.KC0_SPARSE_SIZE ain't fun if u have more than 1 data
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=pf3.INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=pf3.INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=pf3.DOUBLE)
    N = pf3.DOF*len(mesh.nodes)

    #TODO: Element creation based on the connection matrix

    KC0 = ss.coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc() #@# stiffness matrix in sparse format