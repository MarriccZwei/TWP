import sys
sys.path.append('..')

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d import BeamC, BeamCData, BeamCProbe, DOF, INT, DOUBLE

from ...C4.Solution.eleProps import beam_stress_recovery, load_ele_props


def test_static_point_load_square():
    n = 30
    L = 8
    a = .05 #side of the square
    Fx = 10.
    Fy = 5.
    Fz = 2.
    Mx = 4.
    My = 7.
    Mz = 6.

    E = 203.e9 # Pa
    nu = 0.3
    rho = 7.83e3 # kg/m3

    x = np.linspace(0, L, n)
    y = np.ones_like(x)
    hy = a # m
    hz = a # m
    A = hy*hz
    Izz = hz*hy**3/12
    Iyy = hz**3*hy/12
    print('E', E)
    print('Iyy', Iyy)
    print('Izz', Izz)

    ncoords = np.vstack((x, y, np.zeros_like(x))).T
    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))

    n1s = nids[0:-1]
    n2s = nids[1:]

    num_elements = len(n1s)
    print('num_elements', num_elements)

    p = BeamCProbe()
    data = BeamCData()

    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    N = DOF*n
    print('num_DOF', N)

    prop = BeamProp()
    prop.A = A
    prop.E = E
    scf = (5+5*nu)/(6+5*nu)
    prop.G = scf*E/2/(1+nu)
    prop.Izz = Izz
    prop.Iyy = Iyy
    prop.J = Izz + Iyy

    ncoords_flatten = ncoords.flatten()

    #mock usage of element property creation software
    mock_desvars = {
        'W_mb':hy #height of the mock beam
    }
    mock_ele_types = ['mb']*(len(x)-1)
    mock_materials = {
        'NU_ALU':nu,
        'E_ALU':E,
        'SF_ALU':300e6,
        #the below should not be accessed during the test:
        'E_FOAM':None,
        'SF_FOAM':None,
        'NU_FOAM':None,
        'RHO_FOAM':None,
        #this is accessed but it would simply result in multiplications of I, A etc. not worth checking
        'RHO_ALU':2.7e3
    }

    mock_ele_args = [[a]]*(len(x)-1)
    eledict = load_ele_props(mock_desvars, mock_materials, mock_ele_types, mock_ele_args)
    test_beam_props = eledict["beamprops"]
    for tbp in test_beam_props:
        assert np.isclose(tbp.A, prop.A)
        assert np.isclose(tbp.Iyy, prop.Iyy)
        assert np.isclose(tbp.Izz, prop.Izz)
        assert np.isclose(tbp.J, tbp.J)
        assert np.isclose(tbp.E, tbp.E)
        assert np.isclose(tbp.G, tbp.G)

    beams = []
    init_k_KC0 = 0
    for n1, n2 in zip(n1s, n2s):
        pos1 = nid_pos[n1]
        pos2 = nid_pos[n2]
        beam = BeamC(p)
        beam.init_k_KC0 = init_k_KC0
        beam.n1 = n1
        beam.n2 = n2
        beam.c1 = DOF*pos1
        beam.c2 = DOF*pos2
        beam.update_rotation_matrix(1., 1., 0, ncoords_flatten)
        beam.update_probe_xe(ncoords_flatten)
        beam.update_KC0(KC0r, KC0c, KC0v, prop)
        beams.append(beam)
        init_k_KC0 += data.KC0_SPARSE_SIZE

    print('elements created')

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

    print('sparse KC0 created')

    bk = np.zeros(N, dtype=bool)

    left_end = np.isclose(x, 0.)
    right_end = np.isclose(x, L)

    bk[0::DOF][left_end] = True
    bk[1::DOF][left_end] = True
    bk[2::DOF][left_end] = True
    bk[3::DOF][left_end] = True
    bk[4::DOF][left_end] = True
    bk[5::DOF][left_end] = True

    bu = ~bk

    fext = np.zeros(N)
    load_pos = np.isclose(x, L)
    fext[1::DOF][load_pos] = Fx
    fext[1::DOF][load_pos] = Fy
    fext[2::DOF][load_pos] = Fz
    fext[3::DOF][load_pos] = Mx
    fext[4::DOF][load_pos] = My
    fext[5::DOF][load_pos] = Mz

    Kuu = KC0[bu, :][:, bu]

    u = np.zeros(N)
    u[bu] = spsolve(Kuu, fext[bu])

    #stress analytical computation
    #1) internal forces
    #Mx = const, Fx = const, Fy = const, Fz = const
    Myi = lambda xi: My+Fz*(xi-L)
    Mzi = lambda xi: Mz+Fy*(L-xi)

    #2)stress computation at all element centres
    xis = (x[1:]+x[:-1])/2

    #3) at all four corners of the square
    sig_pp = Fx/A-Myi(xis)*hz/2/Izz+Mzi(xis)*hy/2/Iyy
    txz_pp = Fz/A+Mx*hy/2/prop.J
    txy_pp = Fy/A-Mx*hz/2/prop.J
    svm_pp = np.sqrt(sig_pp**2+3*txz_pp**2+3*txy_pp**2)

    sig_np = Fx/A-Myi(xis)*hz/2/Izz-Mzi(xis)*hy/2/Iyy
    txz_np = Fz/A-Mx*hy/2/prop.J
    txy_np = Fy/A-Mx*hz/2/prop.J
    svm_np = np.sqrt(sig_np**2+3*txz_np**2+3*txy_np**2)

    sig_nn = Fx/A+Myi(xis)*hz/2/Izz-Mzi(xis)*hy/2/Iyy
    txz_nn = Fz/A-Mx*hy/2/prop.J
    txy_nn = Fy/A+Mx*hz/2/prop.J
    svm_nn = np.sqrt(sig_nn**2+3*txz_nn**2+3*txy_nn**2)

    sig_pn = Fx/A+Myi(xis)*hz/2/Izz+Mzi(xis)*hy/2/Iyy
    txz_pn = Fz/A+Mx*hy/2/prop.J
    txy_pn = Fy/A+Mx*hz/2/prop.J
    svm_pn = np.sqrt(sig_nn**2+3*txz_nn**2+3*txy_nn**2)

    svm = np.max(np.vstack((svm_pp, svm_pn, svm_nn, svm_np)), axis=0)
    assert svm.shape == svm_pn.shape
    ref_margins = svm/mock_materials["SF_ALU"]

    #stress computation by the software
    margins = list()
    for prop_, ele in zip(test_beam_props, beams):
        ele.update_probe_ue(u)
        ele.update_probe_xe(ncoords_flatten)
        margins.append(beam_stress_recovery(mock_desvars, mock_materials, ele, prop_, eledict['beamorients'][0], 'mb', p)) #all beams should have same orient

    assert np.allclose(margins, ref_margins, rtol=4e-2), f"{margins}, {ref_margins}"

    fint = np.zeros(N)
    for beam in beams:
        beam.update_probe_ue(u)
        beam.update_fint(fint, prop)

    # NOTE adding reaction forces to external force vector
    Kku = KC0[bk, :][:, bu]
    fext[bk] = Kku @ u[bu]
    assert np.allclose(fint, fext)


if __name__ == '__main__':
    test_static_point_load_square()