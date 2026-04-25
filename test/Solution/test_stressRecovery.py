import sys
sys.path.append('..')

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
import pyfe3d.shellprop_utils as psu
from pyfe3d import BeamC, BeamCData, BeamCProbe, DOF, INT, DOUBLE, Quad4Data, Quad4, Quad4Probe

from ...C4.Solution.eleProps import beam_stress_recovery, load_ele_props, quad_stress_recovery
from ...C4.Solution.stressRecovery import recover_stresses, strains_quad


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
    rho = 2.7e3 # kg/m3

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
        'RHO_ALU':rho
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
    svm_pn = np.sqrt(sig_pn**2+3*txz_pn**2+3*txy_pn**2)

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


def test_quad_recovery(sheet_first:bool, plot:bool=False):
    #resolution
    nx = 7
    ny = 30 #must have a middle element

    #dimensions
    lx = .03 #m
    ly = .15 #m
    H = .01 #m

    #mock usage of element property creation software
    nu = .33
    E = 72e9 #Pa
    sf_alu = 100e6 if sheet_first else 1000e6 #Pa
    sf_foam = 1000e6 if sheet_first else 100e6 #Pa
    mock_desvars = {
        'H_sq':H,
        '(2t/H)_sq':2/3
    }
    mock_ele_types = ['sq']*(ny-1)
    mock_materials = {
        'NU_ALU':nu,
        'E_ALU':E,
        'SF_ALU':sf_alu,
        #the 'foam' will have the same stiffness properties as alu, but different strength, so we can judge which part fails
        'E_FOAM':E,
        'SF_FOAM':sf_foam,
        'NU_FOAM':nu,
        #this is accessed but no influence on test
        'RHO_FOAM':2.7e3,
        'RHO_ALU':2.7e3
    }
    mock_ele_args = [[H]]*(ny-1)

    #FEA model
    data = Quad4Data()
    probe = Quad4Probe()

    xtmp = np.linspace(0, lx, nx)
    ytmp = np.linspace(0, ly, ny)
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

    prop = psu.isotropic_plate(H, E, nu)

    quads = []
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

    #checking the quad property creation
    eledict = load_ele_props(mock_desvars, mock_materials, mock_ele_types, mock_ele_args)
    for shellprop in eledict["shellprops"]:
        assert np.isclose(shellprop.A11, prop.A11)
        assert np.isclose(shellprop.A12, prop.A12)
        assert np.isclose(shellprop.A16, prop.A16)
        assert np.isclose(shellprop.A22, prop.A22)
        assert np.isclose(shellprop.A26, prop.A26)
        assert np.isclose(shellprop.A66, prop.A66)
        assert np.isclose(shellprop.B11, prop.B11)
        assert np.isclose(shellprop.B12, prop.B12)
        assert np.isclose(shellprop.B16, prop.B16)
        assert np.isclose(shellprop.B22, prop.B22)
        assert np.isclose(shellprop.B26, prop.B26)
        assert np.isclose(shellprop.B66, prop.B66)
        assert np.isclose(shellprop.D11, prop.D11)
        assert np.isclose(shellprop.D12, prop.D12)
        assert np.isclose(shellprop.D16, prop.D16)
        assert np.isclose(shellprop.D22, prop.D22)
        assert np.isclose(shellprop.D26, prop.D26)
        assert np.isclose(shellprop.D66, prop.D66)
        assert np.isclose(shellprop.E44, prop.E44)
        assert np.isclose(shellprop.E45, prop.E45)
        assert np.isclose(shellprop.E55, prop.E55)

    #loads and BCs, then solution
    bk = np.zeros(N, dtype=bool)
    check = np.isclose(y, 0.)
    for dof in range(DOF):
        bk[dof::DOF] = check
    bu = ~bk

    Ttot = 900.
    Mtot = 10.
    Tper = Ttot/(nx-1)
    Mper = Mtot/(nx-1)
    fext = np.zeros(N)
    check = np.isclose(y, ly)
    fext[1::DOF][check] = Tper #tension
    fext[3::DOF][check] = Mper #bending moment
    check = np.isclose(y, ly) & (np.isclose(x, 0.) | np.isclose(x, lx))
    fext[1::DOF][check] -= Tper/2 #tension
    fext[3::DOF][check] -= Mper/2 #bending moment
    fu = fext[bu]
    assert np.isclose(fu.sum(), Ttot+Mtot)

    KC0uu = KC0[bu, :][:, bu]
    uu = spsolve(KC0uu, fext[bu])
    u = np.zeros(N)
    u[bu] = uu
    w = u[2::DOF].reshape(nx, ny).T

    #plotting solution for stress range
    if plot:
        import matplotlib.pyplot as plt

        plt.gca().set_aspect('equal')
        levels = np.linspace(w.min(), w.max(), 50)
        plt.contourf(xmesh, ymesh, w, levels=levels)
        plt.colorbar()
        plt.show()

    #stress recovery
    for quad in quads:
        yavg = (y[nid_pos[quad.n1]]+y[nid_pos[quad.n2]]+y[nid_pos[quad.n3]]+y[nid_pos[quad.n4]])/4
        if yavg > ly/2:
            quad.update_probe_ue(u)
            quad.update_probe_xe(ncoords_flatten)
            recovered_margin = quad_stress_recovery(mock_desvars, mock_materials, quad, prop, (0, 1, 0), mock_ele_types[0], probe)

            #checking individual stresses
            sbend = 6*Mtot/H**2/lx #at the outermost sheet point
            sbend_foam = 2*Mtot/H**2/lx #at the outermost foam point
            sten = Ttot/H/lx

            strains = strains_quad(probe)
            normal_sheet, shear_sheet, tau_yz_sheet, tau_xz_sheet = recover_stresses(strains, E, nu, shellprop.scf_k13)
            assert np.isclose(normal_sheet(0)[1], sten), f"recovered: {normal_sheet(0)}, reference: {sten}"
            assert np.isclose(normal_sheet(-H/2)[1], sten+sbend, rtol=1e-3), f"recovered: {normal_sheet(-H/2)}, reference: {sten+sbend}"
            assert np.isclose(normal_sheet(-H/6)[1], sten+sbend_foam, rtol=1e-3), f"recovered: {normal_sheet(-H/6)}, reference: {sten+sbend_foam}"
            #NOTE: the shear buildup the closer you are to the rooot, that is due to poisson ratio deformation being constrained there
            #print(f"shears: {tau_xz_sheet}, {tau_yz_sheet}, {shear_sheet(H/2)}")
            
            #for checking sheet failure
            if sheet_first:
                svm = sbend+sten
                ref_margin = svm/sf_alu

                assert np.isclose(recovered_margin, ref_margin, rtol=1e-4), f"recovered: {recovered_margin}, reference: {ref_margin}"

            #for checking foam failure
            else:
                sig = sten+sbend_foam
                sm = sig/3
                svm = sig
                alph = 3*np.sqrt((.5-nu)/(1+nu))
                sfoam = np.sqrt((svm**2+alph**2*sm**2)/(1+(alph/3)**2))
                ref_margin = sfoam/sf_foam
                assert np.isclose(recovered_margin, ref_margin, rtol=1e-4), f"recovered: {recovered_margin}, reference: {ref_margin}"


def test_quad_recovery_x(sheet_first:bool, plot:bool=False):
    #resolution
    nx = 30 #must have a middle element
    ny = 7 

    #dimensions
    lx = .15 #m
    ly = .03 #m
    H = .01 #m

    #mock usage of element property creation software
    nu = .33
    E = 72e9 #Pa
    sf_alu = 100e6 if sheet_first else 1000e6 #Pa
    sf_foam = 1000e6 if sheet_first else 100e6 #Pa
    mock_desvars = {
        'H_sq':H,
        '(2t/H)_sq':2/3
    }
    mock_ele_types = ['sq']*(nx-1)
    mock_materials = {
        'NU_ALU':nu,
        'E_ALU':E,
        'SF_ALU':sf_alu,
        #the 'foam' will have the same stiffness properties as alu, but different strength, so we can judge which part fails
        'E_FOAM':E,
        'SF_FOAM':sf_foam,
        'NU_FOAM':nu,
        #this is accessed but no influence on test
        'RHO_FOAM':2.7e3,
        'RHO_ALU':2.7e3
    }
    mock_ele_args = [[H]]*(nx-1)

    #FEA model
    data = Quad4Data()
    probe = Quad4Probe()

    xtmp = np.linspace(0, lx, nx)
    ytmp = np.linspace(0, ly, ny)
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

    prop = psu.isotropic_plate(H, E, nu)

    quads = []
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

    #checking the quad property creation
    eledict = load_ele_props(mock_desvars, mock_materials, mock_ele_types, mock_ele_args)
    for shellprop in eledict["shellprops"]:
        assert np.isclose(shellprop.A11, prop.A11)
        assert np.isclose(shellprop.A12, prop.A12)
        assert np.isclose(shellprop.A16, prop.A16)
        assert np.isclose(shellprop.A22, prop.A22)
        assert np.isclose(shellprop.A26, prop.A26)
        assert np.isclose(shellprop.A66, prop.A66)
        assert np.isclose(shellprop.B11, prop.B11)
        assert np.isclose(shellprop.B12, prop.B12)
        assert np.isclose(shellprop.B16, prop.B16)
        assert np.isclose(shellprop.B22, prop.B22)
        assert np.isclose(shellprop.B26, prop.B26)
        assert np.isclose(shellprop.B66, prop.B66)
        assert np.isclose(shellprop.D11, prop.D11)
        assert np.isclose(shellprop.D12, prop.D12)
        assert np.isclose(shellprop.D16, prop.D16)
        assert np.isclose(shellprop.D22, prop.D22)
        assert np.isclose(shellprop.D26, prop.D26)
        assert np.isclose(shellprop.D66, prop.D66)
        assert np.isclose(shellprop.E44, prop.E44)
        assert np.isclose(shellprop.E45, prop.E45)
        assert np.isclose(shellprop.E55, prop.E55)

    #loads and BCs, then solution
    bk = np.zeros(N, dtype=bool)
    check = np.isclose(x, 0.)
    for dof in range(DOF):
        bk[dof::DOF] = check
    bu = ~bk

    Ttot = 900.
    Mtot = 10.
    Tper = Ttot/(ny-1)
    Mper = Mtot/(ny-1)
    fext = np.zeros(N)
    check = np.isclose(x, lx)
    fext[0::DOF][check] = Tper #tension
    fext[4::DOF][check] = Mper #bending moment
    check = np.isclose(x, lx) & (np.isclose(y, 0.) | np.isclose(y, ly))
    fext[0::DOF][check] -= Tper/2 #tension
    fext[4::DOF][check] -= Mper/2 #bending moment
    fu = fext[bu]
    assert np.isclose(fu.sum(), Ttot+Mtot)

    KC0uu = KC0[bu, :][:, bu]
    uu = spsolve(KC0uu, fext[bu])
    u = np.zeros(N)
    u[bu] = uu
    w = u[2::DOF].reshape(nx, ny).T

    #plotting solution for stress range
    if plot:
        import matplotlib.pyplot as plt

        plt.gca().set_aspect('equal')
        levels = np.linspace(w.min(), w.max(), 50)
        plt.contourf(xmesh, ymesh, w, levels=levels)
        plt.colorbar()
        plt.show()

    #stress recovery
    for quad in quads:
        xavg = (x[nid_pos[quad.n1]]+x[nid_pos[quad.n2]]+x[nid_pos[quad.n3]]+x[nid_pos[quad.n4]])/4
        if xavg > lx/2:
            quad.update_probe_ue(u)
            quad.update_probe_xe(ncoords_flatten)
            recovered_margin = quad_stress_recovery(mock_desvars, mock_materials, quad, prop, (0, 1, 0), mock_ele_types[0], probe)

            #checking individual stresses
            sbend = 6*Mtot/H**2/ly #at the outermost sheet point
            sbend_foam = 2*Mtot/H**2/ly #at the outermost foam point
            sten = Ttot/H/ly

            strains = strains_quad(probe)
            normal_sheet, shear_sheet, tau_yz_sheet, tau_xz_sheet = recover_stresses(strains, E, nu, shellprop.scf_k13)
            assert np.isclose(normal_sheet(0)[0], sten), f"recovered: {normal_sheet(0)}, reference: {sten}"
            assert np.isclose(normal_sheet(H/2)[0], sten+sbend, rtol=1e-3), f"recovered: {normal_sheet(H/2)}, reference: {sten+sbend}"
            assert np.isclose(normal_sheet(H/6)[0], sten+sbend_foam, rtol=1e-3), f"recovered: {normal_sheet(H/6)}, reference: {sten+sbend_foam}"
            #NOTE: the shear buildup the closer you are to the rooot, that is due to poisson ratio deformation being constrained there
            #print(f"shears: {tau_xz_sheet}, {tau_yz_sheet}, {shear_sheet(H/2)}")
            
            #for checking sheet failure
            if sheet_first:
                svm = sbend+sten
                ref_margin = svm/sf_alu

                assert np.isclose(recovered_margin, ref_margin, rtol=1e-4), f"recovered: {recovered_margin}, reference: {ref_margin}"

            #for checking foam failure
            else:
                sig = sten+sbend_foam
                sm = sig/3
                svm = sig
                alph = 3*np.sqrt((.5-nu)/(1+nu))
                sfoam = np.sqrt((svm**2+alph**2*sm**2)/(1+(alph/3)**2))
                ref_margin = sfoam/sf_foam
                assert np.isclose(recovered_margin, ref_margin, rtol=1e-4), f"recovered: {recovered_margin}, reference: {ref_margin}"


def test_stress_recovery_sub():
    data = Quad4Data()
    probe = Quad4Probe()
    E = 100e9
    nu = .33
    sc = 5/6
    t = .001
    prop = psu.isotropic_plate(t, E, nu)
    KC0r = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE, dtype=DOUBLE)

    quad = Quad4(probe)
    quad.n1 = 1
    quad.n2 = 2
    quad.n3 = 3
    quad.n4 = 4
    quad.c1 = 0
    quad.c2 = 6
    quad.c3 = 12
    quad.c4 = 18
    ncoords_flatten = np.array([0., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0., 0.])
    quad.update_rotation_matrix(ncoords_flatten)
    quad.update_probe_xe(ncoords_flatten)
    quad.update_KC0(KC0r, KC0c, KC0v, prop)

    u = .001 #deformation magnitude

    #case 1: plane shear
    quad.update_probe_ue(np.array([0., 0., 0., 0., 0., 0., 
                                   u, 0., 0., 0., 0., 0., 
                                   u, 0., 0., 0., 0., 0.,
                                   0., 0., 0., 0., 0., 0.]))
    strains = strains_quad(probe)
    _, shear, _, _ = recover_stresses(strains, E, nu, sc)
    ref_shear = u*E/(1+nu)/2
    assert np.isclose(shear(0), ref_shear), f"{shear(0)}, {ref_shear}"

    #case 2: out of plane shear in xz
    quad.update_probe_ue(np.array([0., 0., 0., 0., 0., 0., 
                                   0., 0., 0., 0., 0., 0., 
                                   0., 0., u, 0., 0., 0.,
                                   0., 0., u, 0., 0., 0.]))
    strains = strains_quad(probe)
    _, _, shear, _  = recover_stresses(strains, E, nu, sc)
    ref_shear = u*E/(1+nu)/2*sc
    assert np.isclose(shear, ref_shear), f"{shear}, {ref_shear}"

    #case 3: out of plane shear in yz
    quad.update_probe_ue(np.array([0., 0., 0., 0., 0., 0., 
                                   0., 0., u, 0., 0., 0., 
                                   0., 0., u, 0., 0., 0.,
                                   0., 0., 0., 0., 0., 0.]))
    strains = strains_quad(probe)
    _, _, _, shear  = recover_stresses(strains, E, nu, sc)
    ref_shear = u*E/(1+nu)/2*sc
    assert np.isclose(shear, ref_shear), f"{shear}, {ref_shear}"

    #sign of torsion
    quad.update_probe_ue(np.array([0., 0., 0., 0., 0., 0., 
                                   0., 0., 0., 0., u, 0., 
                                   0., 0., 0., -u, u, 0.,
                                   0., 0., 0., -u, 0., 0.]))
    strains = strains_quad(probe)
    _, shear, _, _  = recover_stresses(strains, E, nu, sc)
    assert(shear(t)>0), shear(t)


if __name__ == '__main__':
    test_static_point_load_square()
    test_quad_recovery(True)
    test_quad_recovery(False)
    test_quad_recovery_x(True)
    test_quad_recovery_x(False)
    test_stress_recovery_sub()