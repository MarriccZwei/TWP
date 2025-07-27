import ptsFromCAD as pfc
import geometricClasses as gcl
import meshingComponents as mc
import pyfe3Dgcl as p3g
import numpy as np
import pyfe3d as pf3
import scipy.sparse.linalg as ssl
import loadModel as lm

import typing as ty
import matplotlib.pyplot as plt
import numpy.typing as nt

def mesh_block(cadData:str, sizerVars:ty.Dict[str,str], eleProps:ty.Dict[str,ty.Dict[str, object]], consts:ty.Dict[str, int], codes:ty.Dict[str,str]):
    up = pfc.UnpackedPoints(cadData)
    mesh = gcl.Mesh3D()
    pts, ids = mc.all_components(mesh, up, consts["NB_COEFF"], consts["NA"], consts["NF2"], consts["NIP_COEFF"], consts["NTRIG"], 
                                 sizerVars["csp"], consts["BAT_MASS_1WING"], consts["M_LE"], 
                                 consts["M_TE"], codes["spar"], codes["panelPlate"], codes["panelRib"], codes["panelFlange"], 
                                 codes["skin"], codes["rail"], consts["M_MOTOR"], consts["MR"], consts["ML"], consts["M_LG"], consts["LGR"], consts["LGL"], consts["M_HINGE"])
    KC0, M, N, x, y, z, outdict = p3g.eles_from_gcl(mesh, eleProps)
    return {'mesh':mesh, 'up':up, 'KC0':KC0, 'M':M, 'N':N, 'x':x, 'y':y, 'z':z, 'pts':pts, 'ids':ids} | outdict

def fem_linear_block(consts:ty.Dict[str, object], meshOuts:ty.Dict[str,object], loadCase:ty.Dict[str,object], ult=False):
    KC0, M, N, x, y, z, mesh, up, ids, pts = tuple(meshOuts[k] for k in ['KC0', 'M', 'N', 'x', 'y', 'z', 'mesh', 'up', 'ids', 'pts'])
    FULLSPAN, MTOM, G0, motorR, motorL= tuple(consts[k] for k in ['FULLSPAN', 'MTOM', 'G0', 'MR', 'ML'])
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
    f[0::pf3.DOF] += f[2::pf3.DOF]/LD #drag points towards positive x!!!

    #applying thrust
    for mtr in up.motors:
        affected_ids = list()
        for i, node in enumerate(mesh.nodes):
            if (node.y >= mtr.y-motorR) and (node.y <= motorR+mtr.y) and (node.x-mtr.x <= motorL/2):
                affected_ids.append(i)
        dFT = FT/len(affected_ids)
        for i in affected_ids:
            f[0::pf3.DOF][i] -= dFT

    #applying landing load
    f[2::pf3.DOF][ids["lg"]] += nlg*G0*MTOM/len(f[2::pf3.DOF][ids["lg"]])

    #applying weight
    weight = p3g.weight(M, nult*G0, N, pf3.DOF, ndir) if ult else p3g.weight(M, n*G0, N, pf3.DOF, ndir)
    f+=weight

    #checks and solution
    KC0uu = KC0[bu, :][:, bu]
    fu = f[bu]
    uu = ssl.spsolve(KC0uu, fu)
    u = np.zeros(N)
    u[bu] = uu

    w = u[2::pf3.DOF]
    v = u[0::pf3.DOF]

    return {'u':u, 'w':w, 'v':v}

def plot_block(w:nt.NDArray, wtxt:str, meshOuts:ty.Dict[str, object], consts:ty.Dict[str, object], plusonly=False,):
    KC0, M, N, x, y, z, mesh, up, ids, pts = tuple(meshOuts[k] for k in ['KC0', 'M', 'N', 'x', 'y', 'z', 'mesh', 'up', 'ids', 'pts'])
    ntrig, nf2= tuple(consts[k] for k in ['NTRIG', 'NF2'])

    fig = plt.figure(figsize=(20,10))
    levels = np.linspace(w.min(), w.max(), 50) if not plusonly else np.linspace(0, w.max(), 50)

    def contourp(loc:int, selection:nt.NDArray[np.int32]):
        plt.subplot(loc) if loc != 0 else ""
        sf = selection.flatten()
        plt.contourf(x[sf].reshape(selection.shape), y[sf].reshape(selection.shape), w[sf].reshape(selection.shape), levels=levels)
        plt.xlabel("x [m]", loc="right")
        plt.ylabel("y [m]", loc="top")

    contourp(231, ids["plateTop"])
    plt.title("Upper Panel")
    contourp(232, ids["skinTop"])
    plt.title("Upper Skin")
    contourp(233, ids["spars"][:, nf2-1:-nf2])
    plt.colorbar()
    for i in ids["sparBends"][1:-1]:
        plt.plot([mesh.nodes[ids["spars"][0, i]].x, mesh.nodes[ids["spars"][-1, i]].x], [mesh.nodes[ids["spars"][0, i]].y, mesh.nodes[ids["spars"][-1, i]].y],
                 color="black", linestyle="dotted", linewidth=.2)
    plt.title("Spars - Without flanges")
    contourp(234, ids["plateBot"])
    plt.title("Lower Panel")
    contourp(235, ids["skinBot"])
    plt.title("Lower Skin")

    # plt.subplot(244)
    # x_ = [0,1,2,3,4,5,6,7,8,9]
    # y_ = [w[ids["motors"][0][0]], w[ids["motors"][0][1]], w[ids["motors"][1][0]], w[ids["motors"][1][1]],
    #     w[ids["motors"][2][0]], w[ids["motors"][2][1]], w[ids["motors"][3][0]], w[ids["motors"][3][1]],
    #     w[ids["lg"]], w[ids["hinge"]]]
    # plt.scatter(x_, y_)
    # plt.xticks(x_, ["m1u","m1l","m2u","m2l", "m3u", "m3l", 
    #                         "m4u", "m4l", "lg", "hn"], rotation=60)
    # plt.ylabel(f"{wtxt} [m]", loc="top")
    # plt.title("Poitmass Components")

    # #TODO: reformat as 9x9, with rails, batteries and LE, TE mass distrs.
    # plt.subplot(247)
    # labels = [f"rail{i}" for i in range(1, 4*ntrig+3)]
    # for batids in ids["bats"]:
    #     x_ = [mesh.nodes[id_].y for id_ in batids]
    #     y_ = [w[id_] for id_ in batids]
    #     plt.plot(x_, y_, label=labels.pop(0))
    # plt.xlabel("y [m]", loc="right")
    # plt.ylabel(f"{wtxt} [m]", loc="top")
    # plt.title("Batteries")
    # plt.legend()
    plt.subplot(236)
    first=True
    for i in range(ids["spars"][:, :nf2].shape[1]):
        plt.plot(y[ids["spars"][:, i]], w[ids["spars"][:, i]], color="pink", label="frontFlange") if first else plt.plot(y[ids["spars"][:, i]], w[ids["spars"][:, i]], color="pink")
        first = False
    first=True
    for i in range(ids["spars"][:, -nf2:].shape[1]):
        plt.plot(y[ids["spars"][:, i]], w[ids["spars"][:, i]], color="green", label="rearFlange") if first else plt.plot(y[ids["spars"][:, i]], w[ids["spars"][:, i]], color="green")
        first = False

    # plt.plot(y[ids["LE"]], w[ids["LE"]], label="LEeqpt")
    # plt.plot(y[ids["TE"]], w[ids["TE"]], label="TEeqpt")
    plt.xlabel("y [m]", loc="right")
    plt.ylabel(f"{wtxt} [m]", loc="top")
    plt.title("Spar Flanges")
    plt.legend()
    fig.subplots_adjust(wspace=0.4)
    return fig


#testing blocks and performing a sensitivity study of springs
if __name__ == "__main__":
    import elementDefs as ed
    import constants as cst
    import copy

    data = "5000;0;-6.5|4750;0;-24|4500;0;-41|4000;0;-75|3500;0;-107|3000;0;-138|2500;0;-167|2000;0;-190|1500;0;-206|1250;0;-211|1000;0;-211.5|750;0;-205|500;0;-187.5|375;0;-173|250;0;-150.5|125;0;-113.5|62.5;0;-82.5|0;0;0|62.5;0;107.5|125;0;149.5|250;0;206.5|375;0;248|500;0;281.5|750;0;330.5|1000;0;363|1250;0;383.5|1500;0;394|2000;0;390|2500;0;362|3000;0;318|3500;0;259|4000;0;187.5|4500;0;104|4750;0;57|5000;0;6.5&4250;18000;2206.872|4125;18000;2198.122|4000;18000;2189.622|3750;18000;2172.622|3500;18000;2156.622|3250;18000;2141.122|3000;18000;2126.622|2750;18000;2115.122|2500;18000;2107.122|2375;18000;2104.622|2250;18000;2104.372|2125;18000;2107.622|2000;18000;2116.372|1937.5;18000;2123.622|1875;18000;2134.872|1812.5;18000;2153.372|1781.25;18000;2168.872|1750;18000;2210.122|1781.25;18000;2263.872|1812.5;18000;2284.872|1875;18000;2313.372|1937.5;18000;2334.122|2000;18000;2350.872|2125;18000;2375.372|2250;18000;2391.622|2375;18000;2401.872|2500;18000;2407.122|2750;18000;2405.122|3000;18000;2391.122|3250;18000;2369.122|3500;18000;2339.622|3750;18000;2303.872|4000;18000;2262.122|4125;18000;2238.622|4250;18000;2213.372&-471.576;3673.46;441.249|126.713;7770.33;945.522|725.002;11867.2;1449.796|1323.292;15964.069;1954.07&3012.266;18950.167;2324.854|4861.588;5721.895;702.56&500;0;0|1999.77;18000;2214.157&3500;0;0|3499.77;18000;2214.157&787.633;1600;66.192|2883.375;1600;66.192|2617.418;1600;526.843|1053.59;1600;526.843&2024.321;18000;2152.592|3273.646;18000;2152.592|3148.758;18000;2368.903|2149.209;18000;2368.903" 
    lgl_infs = [4, 4.5, 5, 6] #preparing the k infinity values
    csts = copy.deepcopy(cst.CONSTS) #not to touch the actual constants dict
    load_case = cst.LOAD_C[2] #landing load case used in the sensitivity study
    load_case["FT"] = 5000 #landing at full thrust - a weird load case that tests everything at once

    for lgl in lgl_infs:
        csts["LGL"] = lgl
        eleDict = ed.eledict(csts, cst.INTIAL, cst.CODES)
        meshOut = mesh_block(data, cst.INTIAL, eleDict, csts, cst.CODES)
        sol = fem_linear_block(csts, meshOut, load_case, True)
        wfig = plot_block(sol['w'], "w", meshOut, csts)
        wfig.savefig(fr"C:\marek\studia\hpb\Results\Sensitivity Study LGL\w\K{lgl}.pdf")
        vfig = plot_block(sol['v'], "v", meshOut, csts)
        vfig.savefig(fr"C:\marek\studia\hpb\Results\Sensitivity Study LGL\v\K{lgl}.pdf")
        print(f"{lgl} is done")