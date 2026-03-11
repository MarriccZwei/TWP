from ..C4.Geometry.geometryInit import geometry_init
from ..C4.Solution.eleProps import load_ele_props
from ..C4.Solution.processLoadCase import process_aeroelastic_load_case
from ..C4.LoadCase import LoadCase
from ..C4.ConfigFiles.classified import MASSES

import aerosandbox as asb 
import pyfe3d as pf3
#import scipy.sparse.linalg as ssl
from pypardiso import spsolve
import numpy as np
import scipy.sparse.linalg as ssl
import aerosandbox as asb

def test_self_weight():
    HYPERPARAMS ={
        'delta':.005,
        'D':.3,
        'd':.02,
        'Delta b':.1,
        '(H/c)_sq':.009,
        '(H/c)_aq':.003,
        '(H/c)_pq':.006
    }

    GEOM_SOURCE ={
        #NOTE: all coordinates in m
        "yfus":1.602374,
        "yhn":18.,
        "ytip":21.,
        "deltazhn":1.362017,
        "deltaxhn":1.548961,
        "(x/c)_fore":.15,
        "(x/c)_rear":.7,
        "cr":5.,
        "ct":2.,
        "ylg":5.768546,
        "deltaxlg":.80115,
        "rlg":.801187,
        "ym1":4.005935,
        "ym2":8.091988,
        "ym3":12.178042,
        "ym4":16.290801,
        "deltaxm1":-.801187,
        "deltaxm2":-.267062,
        "deltaxm3":.267062,
        "deltaxm4":.801187,
        "rootfoil":asb.Airfoil("naca2418"),
        "tipfoil":asb.Airfoil("naca2410")
    }

    N = 5

    model, mesher, excl, wing = geometry_init(GEOM_SOURCE, HYPERPARAMS, MASSES, N, 8)

    desvars = {
        '(2t/H)_sq':0.01,
        '(2t/H)_pq':0.01,
        '(2t/H)_aq':0.01,
        'W_bb':0.003,
        'W_mb':0.003,
        'W_lb':0.003,
        'Ds':0.005
    }

    materials = {
        'E_ALU':72.4e9,
        'NU_ALU':.33,
        'RHO_ALU':2780.,
        'SF_ALU':323.33e6,
        'E_FOAM':6.68e9,
        'NU_FOAM':.275,
        'RHO_FOAM':688.,
        'SF_FOAM':45e6
    }

    ep = load_ele_props(desvars, materials, mesher.eleTypes, mesher.eleArgs)

    beamprops = ep["beamprops"]
    beamorients = ep["beamorients"]
    shellprops = ep["shellprops"]
    matdirs = ep["matdirs"]
    inertia_vals = ep["inertia_vals"]

    model.KC0_M_update(beamprops, beamorients, shellprops, matdirs, inertia_vals)

    print(f"Muu isSym: {(abs(model.Muu-model.Muu.T)>1e-10).nnz == 0}")
    print(f"Kuu isSym: {(abs(model.KC0uu-model.KC0uu.T)>1e-10).nnz == 0}")

    '''Visualise the KA and KC matrices'''
    import matplotlib.pyplot as plt
    # from scipy.sparse import coo_matrix

    # fig = plt.figure()
    # def plot_coo_matrix(m, fig, subplot:int):
    #     if not isinstance(m, coo_matrix):
    #         m = coo_matrix(m)
    #     ax = fig.add_subplot(subplot, facecolor='black')
    #     ax.plot(m.col[m.data>1e-5], m.row[m.data>1e-5], 's', color='green', ms=1)
    #     ax.set_xlim(0, m.shape[1])
    #     ax.set_ylim(0, m.shape[0])
    #     ax.set_aspect('equal')
    #     for spine in ax.spines.values():
    #         spine.set_visible(False)
    #     ax.invert_yaxis()
    #     ax.set_aspect('equal')
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     return ax

    # plot_coo_matrix(abs(model.KC0uu-model.KC0uu.T), fig, 111)
    # plt.show()

    bcoords = model.ncoords[model.bk[0::pf3.DOF]]
    plt.scatter(bcoords[:,0], bcoords[:,2])
    plt.show()

    print("Muu")
    eigminMuu, _ = ssl.eigsh(model.Muu, which="LM", sigma=1e-4)
    print("Muu")
    eigmaxMuu, _ = ssl.eigsh(model.Muu, which="LM")
    print("Kuu")
    eigminKuu, _ = ssl.eigsh(model.KC0uu, which="LM", sigma=100)
    print("Kuu")
    eigmaxKuu, _ = ssl.eigsh(model.KC0uu, which="LM")

    print(f"eigminMuu: {eigminMuu}, eigminKuu: {eigminKuu}")
    print(f"cond M: {abs(max(eigmaxMuu)/min(eigminMuu))}, cond K: {abs(max(eigmaxKuu)/min(eigminKuu))}")


if __name__ == "__main__":
    test_self_weight()
    