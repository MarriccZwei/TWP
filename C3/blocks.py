import ptsFromCAD as pfc
import geometricClasses as gcl
import meshingComponents as mc
import pyfe3Dgcl as p3g
import numpy as np
import pyfe3d as pf3
import scipy.sparse.linalg as ssl
import loadModel as lm

import typing as ty

def mesh_block(cadData:str, sizerVars:ty.Dict[str:str], eleProps:ty.Dict[str,ty.Dict[str, object]], consts:ty.Dict[str, int], codes:ty.Dict[str:str]):
    up = pfc.UnpackedPoints(cadData)
    mesh = gcl.Mesh3D()
    pts, ids = mc.all_components(mesh, up, consts["NB_COEFF"], consts["NA"], consts["NF2"], consts["NIP_COEFF"], consts["NTRIG"], 
                                 consts["DIN"], sizerVars["csp"], sizerVars["bsp"], consts["BAT_MASS_1WING"], consts["M_LE"], 
                                 consts["M_TE"], codes["spar"], codes["panelPlate"], codes["panelRib"], codes["panelFlange"], 
                                 codes["skin"], codes["rail"], codes["varpt"], codes["motor"], codes["lg"], codes["hinge"], codes["mount"])
    KC0, M, N, x, y, z, outdict = p3g.eles_from_gcl(mesh, eleProps)
    return {'mesh':mesh, 'up':up, 'KC0':KC0, 'M':M, 'N':N, 'x':x, 'y':y, 'z':z, 'pts':pts, 'ids':ids} | outdict

def fem_linear_block(consts:ty.Dict[str], meshOuts:ty.Dict[str:object], loadCase:ty.Dict[str:object], ult=False):
    KC0, M, N, x, y, z, mesh, up, ids, pts = tuple(meshOuts[k] for k in ['KC0', 'M', 'N', 'x', 'y', 'z', 'mesh', 'up', 'ids', 'pts'])
    FULLSPAN, MTOM, G0= tuple(consts[k] for k in ['FULLSPAN', 'MTOM', 'G0'])
    n, nult, nlg, ndir, LD, FT  = tuple(loadCase[k] for k in ['n', 'nult', 'nlg', 'ndir', 'LD', 'FT'])
    
    #boundary conditions = fix at y of fuselage boundary
    bk = np.zeros(N, dtype=bool)
    check = np.isclose(y, up.ffb.y)
    for i in range(pf3.DOF):
        bk[i::pf3.DOF] = check
    bu = ~bk

    #loads TODO: add real ones
    f = np.zeros(N)

    #aerodynamic load
    xts, yts, zts = gcl.pts2coords3D(pts["skinTop"].flatten())
    xbs, ybs, zbs = gcl.pts2coords3D(pts["skinBot"].flatten())
    xtmesh, ytmesh = np.array(xts).reshape(pts["skinTop"].shape).T, np.array(yts).reshape(pts["skinTop"].shape).T
    xbmesh, ybmesh = np.array(xbs).reshape(pts["skinBot"].shape).T, np.array(ybs).reshape(pts["skinBot"].shape).T

    #lift fractions - will have to get multiplied by an appropriate load case
    #lt, mxt, myt, ncxp, ncxm, ncyp, ncym, yPerB2, xPerC = lm.apply_on_wingbox(xtmesh, ytmesh, (up.fft.y/FULLSPAN, up.tft.y/FULLSPAN), (up.xcft, up.xcrt), True, True)
    lt, mxt, myt = lm.apply_on_wingbox(xtmesh, ytmesh, (up.fft.y/FULLSPAN, up.tft.y/FULLSPAN), (up.xcft, up.xcrt), True) #(up.fft.y/FULLSPAN, up.tft.y/FULLSPAN), (up.xcft, up.xcrt)
    lb, mxb, myb = lm.apply_on_wingbox(xbmesh, ybmesh, (up.fft.y/FULLSPAN, up.tft.y/FULLSPAN), (up.xcfb, up.xcrb), False) #(up.fft.y/FULLSPAN, up.tft.y/FULLSPAN), (up.xcfb, up.xcrb)

    #for now, just mul by 2.5*76000*G0
    L = n*MTOM*G0 #TODO: resolve nult vs n
    Lt, Mxt, Myt, Lb, Mxb, Myb = L*lt.flatten(), L*mxt.flatten(), L*myb.flatten(), L*lb.flatten(), L*mxb.flatten(), L*myb.flatten()
    #applying the top skin loads
    for id_, Lt_, Mxt_, Myt_ in zip(ids["skinTop"].T.flatten(), Lt, Mxt, Myt):
        f[2::pf3.DOF][id_] = Lt_
        f[3::pf3.DOF][id_] = Mxt_
        f[4::pf3.DOF][id_] = Myt_ 
    #applying the bottom skin loads
    for id_, Lb_, Mxb_, Myb_ in zip(ids["skinBot"].T.flatten(), Lb, Mxb, Myb):
        f[2::pf3.DOF][id_] = Lb_
        f[3::pf3.DOF][id_] = Mxb_
        f[4::pf3.DOF][id_] = Myb_ 

    assert np.isclose(f[2::pf3.DOF].sum(), L/2) #we have to compare with f not fu, cuz in fu part of the lift gets eaten by bce, but that's aight

    #applying drag
    f[0::pf3.DOF] += f[2::pf3.DOF]/LD

    #applying thrust
    for mtr, mid in zip(up.motors, ids["motors"]): #checking coordinate correspondence
        f[0::pf3.DOF][mid[0]] = FT/2
        f[0::pf3.DOF][mid[1]] = FT/2

    #applying landing load
    f[2::pf3.DOF][ids["lg"]] += nlg*G0*MTOM/2

    #applying weight
    weight = p3g.weight(M, nult*G0, N, pf3.DOF, ndir) if ult else p3g.weight(M, n*G0, N, pf3.DOF, ndir)
    f+=weight

    #TODO: Remove this, this is only to check the mopment hypothesis
    # f[3::pf3.DOF] = 0
    # f[4::pf3.DOF] = 0
    # f[5::pf3.DOF] = 0


    #checks and solution
    KC0uu = KC0[bu, :][:, bu]
    fu = f[bu]
    uu = ssl.spsolve(KC0uu, fu)
    u = np.zeros(N)
    u[bu] = uu

    w = u[2::pf3.DOF]
    v = u[0::pf3.DOF]

    return {'u':u, 'w':w, 'v':v}


