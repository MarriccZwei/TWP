from ...C4.Geometry.adaptive_mesh_wall import adaptive_mesh_wall
from ...C4.Geometry.Mesher import Mesher
from ...C4.Pyfe3DModel import Pyfe3DModel

import numpy as np
import matplotlib.pyplot as plt
import pyfe3d as pf3
import pypardiso as ppd
import pyfe3d.shellprop_utils as psu
import scipy.sparse as ss
import scipy.sparse.linalg as ssl

def test_adaptive_mesh(plot=False):
    #geometry definition
    C1 = 5. #m
    C2 = 3. #m
    L = 10. #m
    HPERC = .01
    ip1 = np.array([0., 0., 1.]) #coords in m
    ip2 = np.array([C1, 0., 0.])
    op1 = np.array([1., L, .5])
    op2 = np.array([1.+C2, L, 0.])
    cFun = lambda y: C1*(1-y/L)+C2*(y/L)
    eleArgFun = lambda coords: [HPERC*cFun(coords[:, 1].sum()/4)]

    #resolution settings
    nis = [5, 6, 7, 8, 9, 10, 11, 12]
    nos = [5, 5, 5, 5, 5, 5, 5, 5]
    yss = [
        np.linspace(ip1[1], op1[1], 10),
        np.linspace(ip1[1], op1[1], 15),
        np.linspace(ip1[1], op1[1], 20),
        np.linspace(ip1[1], op1[1], 25),
        np.linspace(ip1[1], op1[1], 30),
        np.linspace(ip1[1], op1[1], 35),
        np.linspace(ip1[1], op1[1], 40),
        np.linspace(ip1[1], op1[1], 45),
    ] 

    #meshing for each resolution
    umaxs:list[float] = list()
    lmults:list[float] = list()
    for ni, no, ys in zip(nis, nos, yss):
        eleCoordLists, eleArgLists = adaptive_mesh_wall(ip1, ip2, op1, op2, 
                                                         ys, ni, no, eleArgFun)
        #mesh creation
        decimals = 5
        mesher = Mesher(5)
        for eleCoords, eleArgs in zip(eleCoordLists, eleArgLists):
            mesher.load_ele(eleCoords, 'sq', eleArgs)
        
        #fem model creation
        ncoords = mesher.nodes
        boundary = lambda x, y, z: tuple(
            [np.isclose(y, 0., atol=.1**decimals)]*pf3.DOF)
        model = Pyfe3DModel(ncoords, boundary)

        #loading elements with their properties
        for eleNodePoses in mesher.eleNodePoses:
            model.load_quad(*eleNodePoses)
        shellprops = [psu.isotropic_plate(
            eleArgs[0], 72e9, .33, rho=2700.) for eleArgs in mesher.eleArgs]
        model.KC0_M_update([], [], shellprops, [(1., 0., 0.)]*len(shellprops), [])

        #applying loads (buckling + bending)
        fshear = 100.
        fcomp = 100.
        check = np.isclose(model.y, L)
        npts = np.count_nonzero(check)
        f = np.zeros(model.N)
        f[1::pf3.DOF][check] = -fcomp/npts
        f[2::pf3.DOF][check] = fshear/npts
        assert np.isclose(f[1::pf3.DOF].sum(), -fcomp)
        assert np.isclose(f[2::pf3.DOF].sum(), fshear)
        fu = f[model.bu]
        
        #solution
        u = np.zeros(model.N)
        u[model.bu] = ppd.spsolve(model.KC0uu, fu)
        umaxs.append(u.max())

        #linear buckling analysis
        KGr = np.zeros(model.sizeKG, dtype=pf3.INT)
        KGc = np.zeros(model.sizeKG, dtype=pf3.INT)
        KGv = np.zeros(model.sizeKG, dtype=pf3.DOUBLE)
        for quad, matdir, shellprop in zip(model.quads, model.matdirs, model.shellprops):
            quad.update_probe_ue(u)
            quad.update_probe_xe(model.ncoords_flatten)
            quad.update_KG(KGr, KGc, KGv, shellprop)
        
        num_eig_lb = 4
        KG = ss.coo_matrix((KGv, (KGr, KGc)), shape=(model.N, model.N)).tocsc()
        KGuu = model.uu_matrix(KG)
        eigvecs = np.zeros((model.N, num_eig_lb))
        PREC = np.max(1/model.KC0uu.diagonal())
        eigvals, eigvecsu = ssl.eigsh(A=PREC*KGuu, k=num_eig_lb, which='SM', M=PREC*model.KC0uu, sigma=1., mode='cayley')
        eigvals = -1./eigvals
        eigvecs[model.bu] = eigvecsu
        lmults.append(eigvals[0])

    print(umaxs)
    print(lmults)
    if plot:
        plt.plot(umaxs)
        plt.plot(lmults)
        plt.show()

        for eleCoords in eleCoordLists:
            plt.plot(eleCoords[:, 0.], eleCoords[:, 1.])
        plt.show()