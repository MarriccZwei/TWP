import pyfe3d.shellprop_utils as psu
import scipy.sparse.linalg as ssl
import pyfe3d as pf3
import pyfe3d.beamprop as pbp
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
    print("quads asserted")

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

def test_beams():
    n = 50
    L = 3

    E = 203.e9 # Pa
    rho = 7.83e3 # kg/m3

    x = np.linspace(0, L, n)
    y = np.ones_like(x)
    b = 0.05 # m
    h = 0.05 # m
    A = h*b
    Izz = b*h**3/12
    Iyy = b**3*h/12
    prop = pbp.BeamProp()
    prop.A = A
    prop.E = E
    scf = 5/6.
    prop.G = scf*E/2/(1+0.3)
    prop.Izz = Izz
    prop.Iyy = Iyy
    prop.intrho = rho*A
    prop.intrhoy2 = rho*Izz
    prop.intrhoz2 = rho*Iyy
    prop.J = Izz + Iyy

    ncoords = np.vstack((x, y, np.zeros_like(x))).T
    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))

    n1s = nids[0:-1]
    n2s = nids[1:]

    num_elements = len(n1s)
    print('num_elements', num_elements)

    #adjusting to the poses
    pos1s = [nid_pos[n1] for n1 in n1s]
    pos2s = [nid_pos[n2] for n2 in n2s]

    #model creation
    def boundary(x,y,z):
        is_bk = np.isclose(x, 0)
        return (is_bk, is_bk, is_bk ,is_bk, is_bk, is_bk)
    
    print("boundary")
    model = Pyfe3DModel(ncoords=ncoords, boundary=boundary)
    for pos1, pos2 in zip(pos1s, pos2s):
        model.load_beam(pos1, pos2)
    
    model.KC0_M_update([prop]*num_elements, [(0,1,1)]*num_elements, [], [], [])

    #comparison with the original model
    data = pf3.BeamCData()
    probe = pf3.BeamCProbe()
    ncoords_flatten =ncoords.flatten()

    print("old_model")

    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=pf3.INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=pf3.INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=pf3.DOUBLE)
    Mr = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=pf3.INT)
    Mc = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=pf3.INT)
    Mv = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=pf3.DOUBLE)
    N = pf3.DOF*n

    print("matrices")

    beams = []
    init_k_KC0 = 0
    init_k_M = 0
    for n1, n2 in zip(n1s, n2s):
        pos1 = nid_pos[n1]
        pos2 = nid_pos[n2]
        beam = pf3.BeamC(probe)
        beam.init_k_KC0 = init_k_KC0
        beam.init_k_M = init_k_M
        beam.n1 = n1
        beam.n2 = n2
        beam.c1 = pf3.DOF*pos1
        beam.c2 = pf3.DOF*pos2
        beam.update_rotation_matrix(1., 1., 0, ncoords_flatten)
        beam.update_probe_xe(ncoords_flatten)
        beam.update_KC0(KC0r, KC0c, KC0v, prop)
        beam.update_M(Mr, Mc, Mv, prop)
        beams.append(beam)
        init_k_KC0 += data.KC0_SPARSE_SIZE
        init_k_M += data.M_SPARSE_SIZE

    print("correct elements created")

    for beam, beamcorrect in zip(model.beams, beams):
        assert beam.n1==beamcorrect.n1, f"n1 {beam.n1}, {beamcorrect.n1}"
        assert beam.n1==beamcorrect.n1, f"n1 {beam.n2}, {beamcorrect.n2}"
        assert beam.c1==beamcorrect.c1, f"n1 {beam.c1}, {beamcorrect.c1}"
        assert beam.c2==beamcorrect.c2, f"n1 {beam.c2}, {beamcorrect.c2}"
        assert beam.init_k_KC0==beamcorrect.init_k_KC0
        assert beam.init_k_M==beamcorrect.init_k_M
    for i in range(len(KC0v)):
        assert np.isclose(KC0v[i], model.KC0v[i], atol=.001, rtol=.001), f"{KC0v[i]}, {model.KC0v[i]}, i: {i}"


    #solution testing
    num_eigenvalues = 6
    eigvals, eigvecsu = ssl.eigsh(A=model.KC0uu, M=model.Muu, sigma=-1., which='LM',
            k=num_eigenvalues, tol=1e-4)
    omegan = eigvals**0.5

    alpha123 = np.array([1.875, 4.694, 7.885])
    omega123 = alpha123**2*np.sqrt(E*Izz/(rho*A*L**4))
    print('Theoretical omega123', omega123)
    print('Numerical omega123', omegan)
    print()
    assert np.allclose(np.repeat(omega123, 2), omegan, rtol=0.015)


def test_inertia():
    ncoords = np.array([[1,2,3], [4,5,6], [5,6,7]])
    model = Pyfe3DModel(ncoords, lambda x,y,z:(np.isclose(x,1),np.isclose(x,1),np.isclose(x,1),True,True,True))
    model.load_inertia(0)
    model.load_inertia(2)
    model.KC0_M_update([], [], [], [], [(2.,1.,3.,7.), (6.,9.,9.,7.)])
    assert np.allclose([2,2,2,1,3,7,0,0,0,0,0,0,6,6,6,9,9,7], model.M.diagonal())
    assert np.allclose([0,0,0,6,6,6], model.Muu.diagonal())
    

if __name__ == "__main__":
    PLOT=True
    test_beams()
    test_inertia()
    test_quads()