import pyfe3d.shellprop_utils as psu
import scipy.sparse.linalg as ssl
import pyfe3d as pf3
import pyfe3d.shellprop_utils as psu
import numpy as np

from ..C4.Pyfe3DModel import Pyfe3DModel

PLOT=True
    
def test_quads():
    """
    testing if returs the same results as the isotropic quad test from pyfe3D
    """

    nx = 7
    ny = 11

    a = 3
    b = 7
    h = 0.005 # m

    E = 200e9
    nu = 0.3
    rho = 200

    xtmp = np.linspace(0, a, nx)
    ytmp = np.linspace(0, b, ny)
    xmesh, ymesh = np.meshgrid(xtmp, ytmp)
    ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten(), np.zeros_like(ymesh.T.flatten()))).T

    prop = psu.isotropic_plate(thickness=h, E=E, nu=nu, calc_scf=True, rho=rho)
    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))
    nids_mesh = nids.reshape(nx, ny)
    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()

    num_elements = len(n1s)

    #converting to pos-based ids
    pos1 = np.array([nid_pos[n1] for n1 in n1s])
    pos2 = np.array([nid_pos[n2] for n2 in n2s])
    pos3 = np.array([nid_pos[n3] for n3 in n3s])
    pos4 = np.array([nid_pos[n4] for n4 in n4s])

    #initializing the model
    model = Pyfe3DModel(ncoords, 
                        lambda x,y,z: (True, True, np.isclose(x, 0.) | np.isclose(x, a) | np.isclose(y, 0) | np.isclose(y, b),
                                    False, False, False))
    
    for p1, p2, p3, p4 in zip(pos1, pos2, pos3, pos4):
        model.load_quad(p1, p2, p3, p4)
    
    model.KC0_M_update([], [], [prop]*num_elements, [(1,0,0)]*num_elements, [])

    #load application
    f = np.zeros(model.N)
    fmid = 1.
    check = np.isclose(model.x, a/2) & np.isclose(model.y, b/2)
    f[2::pf3.DOF][check] = fmid #applying the point loads on the node close to the middle
    fu = f[model.bu]
    assert fu.sum() == fmid

    #NOTE: check on bk
    bk = np.zeros(model.N, dtype=bool)
    check = np.isclose(model.x, 0.) | np.isclose(model.x, a) | np.isclose(model.y, 0) | np.isclose(model.y, b)
    bk[2::pf3.DOF] = check

    #constraining the displacements k-known DOF ~k <=> u - unknown DOF
    bk[0::pf3.DOF] = True
    bk[1::pf3.DOF] = True

    bu = ~bk
    assert np.allclose(bu, model.bu)

    #NOTE: check on element connectivity
    quads = []
    assert(np.allclose(model.ncoords_flatten, ncoords.flatten()))
    init_k_KC0 = 0
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        pos1 = nid_pos[n1]
        pos2 = nid_pos[n2]
        pos3 = nid_pos[n3]
        pos4 = nid_pos[n4]
        r1 = ncoords[pos1]
        r2 = ncoords[pos2]
        r3 = ncoords[pos3]
        normal = np.cross(r2 - r1, r3 - r2)[2]
        assert normal > 0
        quad = pf3.Quad4(pf3.Quad4Probe())
        quad.n1 = n1 #@# connectring it to the global stiffness matrix
        quad.n2 = n2
        quad.n3 = n3
        quad.n4 = n4
        quad.c1 = pf3.DOF*nid_pos[n1]
        quad.c2 = pf3.DOF*nid_pos[n2]
        quad.c3 = pf3.DOF*nid_pos[n3]
        quad.c4 = pf3.DOF*nid_pos[n4]
        quad.init_k_KC0 = init_k_KC0
        quad.update_rotation_matrix(model.ncoords_flatten)
        quad.update_probe_xe(model.ncoords_flatten)
        #quad.update_KC0(KC0r, KC0c, KC0v, prop) #matrix contribution, changing the matrices sent
        quads.append(quad)
        init_k_KC0 += pf3.Quad4Data().KC0_SPARSE_SIZE

    for qm, qcorrect, in zip(model.quads, quads):
        assert qm.n1 == qcorrect.n1
        assert qm.n2 == qcorrect.n2
        assert qm.n3 == qcorrect.n3
        assert qm.n4 == qcorrect.n4
        #assert qm.c1 == qcorrect.c1
        #assert qm.c2 == qcorrect.c2
        #assert qm.c3 == qcorrect.c3
        #assert qm.c4 == qcorrect.c4
        print("quad asserted")

    #@# actually solving the fem model
    uu, info = ssl.cg(model.KC0uu, fu, atol=1e-9)
    assert info == 0

    u = np.zeros(model.N)
    u[model.bu] = uu

    w = u[2::pf3.DOF].reshape(nx, ny).T #@# reformatting the result

    # obtained with bfsplate2d element, nx=ny=29
    wmax_ref = 6.594931610258557e-05
    # obtained with Quad4 nx=7, ny=11
    wmax_ref = 6.496928101916171e-05
    print('w.max()', w.max())
    assert np.isclose(wmax_ref, w.max(), rtol=0.02)
    if PLOT:
        import matplotlib.pyplot as plt
        plt.gca().set_aspect('equal')
        levels = np.linspace(w.min(), w.max(), 10)
        plt.contourf(xmesh, ymesh, w, levels=levels)
        plt.colorbar()
        plt.show()

if __name__ == "__main__":
    PLOT=True
    test_quads()