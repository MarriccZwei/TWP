from ...C4.Solution.stressRecovery import *

import sys
sys.path.append('..')

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Quad4, Quad4Data, Quad4Probe, INT, DOUBLE, DOF


def test_static_plate_quad_point_load(plot=False):
    data = Quad4Data()
    probe = Quad4Probe()
    nx = 26
    ny = 11

    a = 13
    b = 3
    h = 0.005 # m

    E = 200e9
    nu = 0.3

    xtmp = np.linspace(0, a, nx)
    ytmp = np.linspace(0, b, ny)
    xmesh, ymesh = np.meshgrid(xtmp, ytmp)
    ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten(), np.zeros_like(ymesh.T.flatten()))).T
    x = ncoords[:, 0]
    y = ncoords[:, 1]
    z = ncoords[:, 2]
    ncoords_flatten = ncoords.flatten()

    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))
    nids_mesh = nids.reshape(nx, ny)
    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()

    num_elements = len(n1s)

    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    N = DOF*nx*ny

    prop = isotropic_plate(thickness=h, E=E, nu=nu, calc_scf=True)

    quads = []
    init_k_KC0 = 0
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        pos1 = nid_pos[n1]
        pos2 = nid_pos[n2]
        pos3 = nid_pos[n3]
        r1 = ncoords[pos1]
        r2 = ncoords[pos2]
        r3 = ncoords[pos3]
        normal = np.cross(r2 - r1, r3 - r2)[2]
        assert normal > 0
        quad = Quad4(probe)
        quad.n1 = n1
        quad.n2 = n2
        quad.n3 = n3
        quad.n4 = n4
        quad.c1 = DOF*nid_pos[n1]
        quad.c2 = DOF*nid_pos[n2]
        quad.c3 = DOF*nid_pos[n3]
        quad.c4 = DOF*nid_pos[n4]
        quad.init_k_KC0 = init_k_KC0
        quad.update_rotation_matrix(ncoords_flatten)
        quad.update_probe_xe(ncoords_flatten)
        quad.update_KC0(KC0r, KC0c, KC0v, prop)
        quads.append(quad)
        init_k_KC0 += data.KC0_SPARSE_SIZE

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

    print('elements created')

    bk = np.zeros(N, dtype=bool)
    check = np.isclose(x, 0.) #| np.isclose(x, a) | np.isclose(y, 0) | np.isclose(y, b)
    bk[0::DOF] = check
    bk[1::DOF] = check
    bk[2::DOF] = check
    bk[3::DOF] = check
    bk[4::DOF] = check
    bk[5::DOF] = check

    bu = ~bk

    # point load at center node
    fext = np.zeros(N)
    fmid = 1.
    check = np.isclose(x, a)
    fext[0::DOF][check] = fmid/(ny-1)
    check = np.isclose(x, a) & (np.isclose(y,b) | np.isclose(y,0))
    fext[0::DOF][check] /= 2

    KC0uu = KC0[bu, :][:, bu]
    assert np.isclose(fext[bu].sum(), fmid)

    uu = spsolve(KC0uu, fext[bu])

    u = np.zeros(N)
    u[bu] = uu

    x2plot = np.zeros(num_elements)
    y2plot = np.zeros(num_elements)
    Nxx2plot = np.zeros(num_elements)
    
    fint = np.zeros(N)
    for i, quad in enumerate(quads):
        quad.update_probe_xe(ncoords_flatten)
        quad.update_probe_ue(u)
        quad.update_fint(fint, prop)
        interpols = strains_resultants_quad(probe, prop)
        x2plot[i] = x[nid_pos[quad.n1]]
        y2plot[i] = y[nid_pos[quad.n1]]
        Nxx2plot[i]= interpols["Nxx"](0, 0)
        print(Nxx2plot[i])
        

    print("Stress recovery tests passed.")

    # NOTE adding reaction forces to external force vector
    Kku = KC0[bk, :][:, bu]
    fext[bk] = Kku @ u[bu]
    atol = 1e-5
    tmp = np.where(np.logical_not(np.isclose(fint, fext, atol=atol)))
    print(tmp)
    print(fint[tmp[0]])
    print(fext[tmp[0]])
    print((KC0@u)[tmp[0]])
    assert np.allclose(fint, fext, atol=atol)

    if plot:
        import matplotlib.pyplot as plt

        plt.subplot(3,3,1)
        ux = u[0::DOF].reshape(nx,ny).T
        levels = np.linspace(ux.min(), ux.max(), 10)
        plt.contourf(xmesh, ymesh, u[0::DOF].reshape(nx,ny).T, levels=levels)
        plt.colorbar()

        plt.subplot(3,3,4)
        levels = np.linspace(Nxx2plot.min(), Nxx2plot.max(), 10)
        plt.contourf(x2plot.reshape((nx-1, ny-1)).T, y2plot.reshape((nx-1, ny-1)).T, Nxx2plot.reshape((nx-1, ny-1)).T, levels=levels)
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    test_static_plate_quad_point_load(plot=True)