from ...C4.Solution.stressRecovery import *

import sys
sys.path.append('..')

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
import typing as ty

from pyfe3d.shellprop_utils import laminated_plate
from pyfe3d import Quad4, Quad4Data, Quad4Probe, INT, DOUBLE, DOF


def test_static_plate_quad_point_load(xdim:float, ydim:float, loads2apply:ty.List[int], plot=False):
    data = Quad4Data()
    probe = Quad4Probe()
    nx = 11*(int(xdim/ydim/4)+1)
    ny = 11*(int(ydim/xdim/4)+1)

    a = xdim
    b = ydim
    h = 0.02 # m

    E = 200e9
    nu = 0.3

    #sandwich data
    Ec = 20e9
    nuc = .5
    hc = .019 # m
    hs = (h-hc)/2

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

    prop = laminated_plate([0,0,0], laminaprops=[(E, nu), (Ec, nuc), (E, nu)], plyts=[hs, hc, hs])

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
    check = np.isclose(x, 0.) if xdim>ydim else np.isclose(y, 0.)
    bk[0::DOF] = check
    bk[1::DOF] = check
    bk[2::DOF] = check
    bk[3::DOF] = check
    bk[4::DOF] = check
    bk[5::DOF] = check

    bu = ~bk

    #load application
    fext = np.zeros(N)
    fmid = 1.

    checkx = np.isclose(x, a)
    checky = np.isclose(y, b)

    def apply_load(check, idx, n):
        fext[idx::pf3.DOF][check] = fmid/n

    if 0 in loads2apply: #Nxx
        apply_load(checkx, 0, ny)
    if 1 in loads2apply: #Nyy
        apply_load(checky, 1, nx)
    if 2 in loads2apply: #Nxy
        apply_load(checky, 0, nx)
    if 3 in loads2apply: #Mxx
        apply_load(checky, 3, nx)
    if 4 in loads2apply: #Myy
        apply_load(checkx, 4, ny)
    if 5 in loads2apply: #Mxy
        apply_load(checky, 4, nx)
    if 7 in loads2apply: #Qxz
        apply_load(checky, 2, nx)
    if 6 in loads2apply: #Qyz
        apply_load(checkx, 2, ny)


    KC0uu = KC0[bu, :][:, bu]
    assert np.isclose(fext[bu].sum(), fmid*len(loads2apply))

    uu = spsolve(KC0uu, fext[bu])

    u = np.zeros(N)
    u[bu] = uu

    exx, eyy, gxy, kxx, kyy, kxy, gyz, gxz, x2plot, y2plot = (np.zeros(num_elements), np.zeros(num_elements), 
    np.zeros(num_elements), np.zeros(num_elements), np.zeros(num_elements), np.zeros(num_elements), np.zeros(num_elements),
    np.zeros(num_elements), np.zeros(num_elements), np.zeros(num_elements))

    Nxx, Nyy,Nxy, Mxx, Myy, Mxy, Qxz, Qyz = (np.zeros(num_elements), np.zeros(num_elements), 
    np.zeros(num_elements), np.zeros(num_elements), np.zeros(num_elements), np.zeros(num_elements), np.zeros(num_elements),
    np.zeros(num_elements))

    sheetR, coreR, sheetR_gt, coreR_gt = (np.zeros(num_elements), np.zeros(num_elements), 
    np.zeros(num_elements), np.zeros(num_elements))
    
    ABDE = get_ABDE(prop)

    fint = np.zeros(N)
    for i, quad in enumerate(quads):
        quad.update_probe_xe(ncoords_flatten)
        quad.update_probe_ue(u)
        quad.update_fint(fint, prop)
        strains = strains_quad(probe)
        resultants = ABDE@strains
        exx[i] = strains[0]
        eyy[i] = strains[1]
        gxy[i] = strains[2]
        kxx[i] = strains[3]
        kyy[i] = strains[4]
        kxy[i] = strains[5]
        gxz[i] = strains[6]
        gyz[i] = strains[7]
        x2plot[i] = x[nid_pos[quad.n1]]
        y2plot[i] = y[nid_pos[quad.n2]]
        Nxx[i] = resultants[0]
        Nyy[i] = resultants[1]
        Nxy[i] = resultants[2]
        Mxx[i] = resultants[3]
        Myy[i] = resultants[4]
        Mxy[i] = resultants[5]
        Qxz[i] = resultants[6]
        Qyz[i] = resultants[7]
        coreR[i], sheetR[i] = sandwich_recovery(probe, [hc/2, h/2], [Ec,E], [nuc, nu], prop.scf_k13)

        #Gt computation
        if not 2 in loads2apply:
            #NOTE: gt does not account for saint venant - compare far from boundaries
            Mgt_i = (a-x2plot[i])/b if a>b else (b-y2plot[i])/a
            Ngt_i = 1/b if a>b else 1/a
            #Qgt_i = Ngt_i

            Ns = Ngt_i/(1+Ec/E)
            Ms = Mgt_i/(1+Ec*hc**3/6/E/(hc+hs)**2/hs)
            Nc = Ngt_i/(1+E/Ec)
            Mc = Mgt_i/(1+E/hc**3*6/Ec*(hc+hs)**2*hs)

            sheetR_gt[i] = (.25*(Ms/hs/(hs+hc)+Ns/2/hs)**2+(Ns/2/hs)**2)**.5
            coreR_gt[i] = (.25*(6*Mc/hc**2+Nc/hc)**2+(Nc/hc)**2)**.5
        else:
            Qs = Nxy[i]/(1+Ec/E)
            Ts = Mxy[i]/(1+Ec*hc**3/6/E/(hc+hs)**2/hs)
            Ns = (Nyy[i]-Nxx[i])/(1+Ec/E) #what goes into mohr
            Qc = Nxy[i]/(1+E/Ec)
            Tc = Mxy[i]/(1+E/hc**3*6/Ec*(hc+hs)**2*hs)
            Nc = (Nyy[i]-Nxx[i])/(1+E/Ec)

            sheetR_gt[i] = (.25*(Ns/2/hs)**2+(Qs/2/hs+Ts/hs/(hs+hc))**2)**.5
            coreR_gt[i] = (.25*(Nc/hc)**2+(Qc/hc+6*Tc/hc**2)**2)**.5



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

    if len(plot)>0:
        import matplotlib.pyplot as plt

        def resultant_plot(i, qty, title, rows=2, cols=4):
            ax = plt.subplot(rows,cols,i)
            levels = np.linspace(qty.min(), qty.max(), 50)
            plt.contourf(x2plot.reshape((nx-1, ny-1)).T, 
                         y2plot.reshape((nx-1, ny-1)).T, 
                         qty.reshape((nx-1, ny-1)).T, levels=levels)
            plt.colorbar()
            ax.set_title(title)

        if 0 in plot:
            resultant_plot(1, exx, "exx")
            resultant_plot(2, eyy, "eyy")
            resultant_plot(3, gxy, "gxy")
            resultant_plot(4, gxz, "gxz")
            resultant_plot(8, gyz, "gyz")
            resultant_plot(5, kxx, "kxx")
            resultant_plot(6, kyy, "kyy")
            resultant_plot(7, kxy, "kxy")

            plt.subplots_adjust(wspace=.5)
            plt.show()

        if 1 in plot:
            resultant_plot(1, Nxx, "Nxx")
            resultant_plot(2, Nyy, "Nyy")
            resultant_plot(3, Nxy, "Nxy")
            resultant_plot(4, Qxz, "Qxz")
            resultant_plot(8, Qyz, "Qyz")
            resultant_plot(5, Mxx, "Mxx")
            resultant_plot(6, Myy, "Myy")
            resultant_plot(7, Mxy, "Mxy")

            plt.subplots_adjust(wspace=.5)
            plt.show()

        if 2 in plot:
            resultant_plot(1, coreR, "core R", 2, 2)
            resultant_plot(2, sheetR, "sheet R", 2, 2)
            resultant_plot(3, coreR_gt, "core GT", 2, 2)
            resultant_plot(4, sheetR_gt, "sheet GT", 2, 2)
            plt.show()


if __name__ == '__main__':
    #plot is empty => don't plot plot has 0 plot strains, plot has 1 plot resultants, plot has 2 - plot radii
    test_static_plate_quad_point_load(1, .1, [0,6], plot=[2])
    test_static_plate_quad_point_load(.1, 1, [1,7], plot=[2])
    test_static_plate_quad_point_load(.1, 1, [2,5], plot=[2])