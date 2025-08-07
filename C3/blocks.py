import ptsFromCAD as pfc
import geometricClasses as gcl
import meshingComponents as mc
import pyfe3Dgcl as p3g
import numpy as np
import pyfe3d as pf3
import scipy.sparse.linalg as ssl
import pypardiso as ppd
import loads as ls

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

    #preparing parmeters for the aerodynamic model
    les = [up.outleline.for_y(y_) for y_ in consts["FOIL_YS"]]
    tes = [up.outteline.for_y(y_) for y_ in consts["FOIL_YS"]]
    return {'mesh':mesh, 'up':up, 'KC0':KC0, 'M':M, 'N':N, 'x':x, 'y':y, 'z':z, 'pts':pts, 'ids':ids,
            'les':les, 'tes':tes} | outdict

def fem_linear_block(consts:ty.Dict[str, object], meshOuts:ty.Dict[str,object], loadCase:ty.Dict[str,object], ult=False, debug=False):
    KC0, M, N, x, y, z, mesh, up, ids, pts, les, tes, ncoords = tuple(meshOuts[k] for k in ['KC0', 'M', 'N', 'x', 'y', 'z', 'mesh', 'up', 'ids', 'pts', 'les', 'tes', 'ncoords'])
    FULLSPAN, MTOM, G0, motorR, motorL, foils, bres, cres= tuple(consts[k] for k in ['FULLSPAN', 'MTOM', 'G0', 'MR', 'ML', 'FOILS', 'BRES', 'CRES'])
    n, nult, nlg, ndir, LD, FT, op  = tuple(loadCase[k] for k in ['n', 'nult', 'nlg', 'ndir', 'LD', 'FT', 'op'])
    
    #boundary conditions = fix at y of fuselage boundary
    bk = np.zeros(N, dtype=bool)
    check = np.isclose(y, up.ffb.y)
    for i in range(pf3.DOF):
        bk[i::pf3.DOF] = check
    bu = ~bk

    #loads
    f = np.zeros(N)

    #aerodynamic forces
    #airplane, vlm, forces, moments = ls.vlm(les, tes, foils, op)
    ids_s = np.hstack([ids["plateTop"][:, -1].flatten(), ids["skinTop"].flatten(), ids["plateTop"][:, 0].flatten(),
                        ids["plateBot"][:, 0].flatten(), ids["skinBot"].flatten(), ids["plateBot"][:, -1].flatten()])
    ncoords_s = ncoords[ids_s, :]
    W, Fext, W_u_to_p, dFv_dp, KA, vlm = ls.KA(ncoords_s, ids_s, N, pf3.DOF, les, tes, foils, op, up.fft.y, bres, cres)
    f += Fext

    if debug:
        vortex_valid = vlm.vortex_centers[:, 1] > ncoords_s[:, 1].min()
        Fv = vlm.forces_geometry[vortex_valid].flatten()
        Mx_v = (vlm.vortex_centers[vortex_valid][:, 1]*Fv[2::3]).sum()
        Mx_u = (ncoords[:, 1]*Fext[2::pf3.DOF]).sum()
        print(Mx_v, Mx_u, ' error:', Mx_u/Mx_v - 1)

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
    wdir = gcl.Direction3D(-ndir.x, -ndir.y, -ndir.z)
    weight = p3g.weight(M, nult*G0, N, pf3.DOF, wdir) if ult else p3g.weight(M, n*G0, N, pf3.DOF, wdir)
    f+=weight

    #checks and solution
    KC0uu = KC0[bu, :][:, bu]
    KAuu = KA[bu, :][:, bu]
    fu = f[bu]
    uu = ppd.spsolve(KC0uu-KAuu, fu)
    u = np.zeros(N)
    u[bu] = uu

    w = u[2::pf3.DOF]
    v = u[0::pf3.DOF]

    return {'u':u, 'w':w, 'v':v, 'bu':bu, 'bk':bk, 'KC0uu':KC0uu, 'W':W, 'ncoords_s':ncoords_s, 'ids_s':ids_s}

def post_processor_block(defl:ty.Dict[str, nt.NDArray[np.float64]], meshOuts:ty.Dict[str,object]):
    eleDict = meshOuts["elements"]
    fi = np.zeros(meshOuts["N"])
    #only for quads and beams as of now - that's all we are using
    for quad in eleDict["quad"]:
        quad.update_probe_finte(quad.shellprop)
        quad.update_fint(fi, quad.shellprop)
    for beam in eleDict["beam"]:
        beam.update_probe_finte(beam.beamprop)
        beam.update_fint(fi, beam.beamprop)

    return {"elements":eleDict, "fi":fi}

def plot_block(w:nt.NDArray, wtxt:str, meshOuts:ty.Dict[str, object], consts:ty.Dict[str, object], unit="m", plusonly=False,):
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

    plt.subplot(236)
    first=True
    for i in range(ids["spars"][:, :nf2].shape[1]):
        plt.plot(y[ids["spars"][:, i]], w[ids["spars"][:, i]], color="pink", label="frontFlange") if first else plt.plot(y[ids["spars"][:, i]], w[ids["spars"][:, i]], color="pink")
        first = False
    first=True
    for i in range(ids["spars"][:, -nf2:].shape[1]):
        plt.plot(y[ids["spars"][:, -i-1]], w[ids["spars"][:, -i-1]], color="green", label="rearFlange") if first else plt.plot(y[ids["spars"][:, -i-1]], w[ids["spars"][:, -i-1]], color="green")
        first = False

    # plt.plot(y[ids["LE"]], w[ids["LE"]], label="LEeqpt")
    # plt.plot(y[ids["TE"]], w[ids["TE"]], label="TEeqpt")
    plt.xlabel("y [m]", loc="right")
    plt.ylabel(f"{wtxt} [{unit}]", loc="top")
    plt.title("Spar Flanges")
    plt.legend()
    fig.subplots_adjust(wspace=0.4)
    return fig


#testing blocks and performing a sensitivity study of springs
if __name__ == "__main__":
    import elementDefs as ed
    import constants as cst
    import copy

    data = "5000;0;-6.5|4750;0;-24|4500;0;-41|4000;0;-75|3500;0;-107|3000;0;-138|2500;0;-167|2000;0;-190|1500;0;-206|1250;0;-211|1000;0;-211.5|750;0;-205|500;0;-187.5|375;0;-173|250;0;-150.5|125;0;-113.5|62.5;0;-82.5|0;0;0|62.5;0;107.5|125;0;149.5|250;0;206.5|375;0;248|500;0;281.5|750;0;330.5|1000;0;363|1250;0;383.5|1500;0;394|2000;0;390|2500;0;362|3000;0;318|3500;0;259|4000;0;187.5|4500;0;104|4750;0;57|5000;0;6.5&4250;18000;2206.872|4125;18000;2198.122|4000;18000;2189.622|3750;18000;2172.622|3500;18000;2156.622|3250;18000;2141.122|3000;18000;2126.622|2750;18000;2115.122|2500;18000;2107.122|2375;18000;2104.622|2250;18000;2104.372|2125;18000;2107.622|2000;18000;2116.372|1937.5;18000;2123.622|1875;18000;2134.872|1812.5;18000;2153.372|1781.25;18000;2168.872|1750;18000;2210.122|1781.25;18000;2263.872|1812.5;18000;2284.872|1875;18000;2313.372|1937.5;18000;2334.122|2000;18000;2350.872|2125;18000;2375.372|2250;18000;2391.622|2375;18000;2401.872|2500;18000;2407.122|2750;18000;2405.122|3000;18000;2391.122|3250;18000;2369.122|3500;18000;2339.622|3750;18000;2303.872|4000;18000;2262.122|4125;18000;2238.622|4250;18000;2213.372&-471.576;3673.46;441.249|126.713;7770.33;945.522|725.002;11867.2;1449.796|1323.292;15964.069;1954.07&3012.266;18950.167;2324.854|4861.588;5721.895;702.56&500;0;0|1999.77;18000;2214.157&3500;0;0|3499.77;18000;2214.157&787.633;1600;66.192|2883.375;1600;66.192|2617.418;1600;526.843|1053.59;1600;526.843&2024.321;18000;2152.592|3273.646;18000;2152.592|3148.758;18000;2368.903|2149.209;18000;2368.903&0;0;0|5000;0;0|1749.77;18000;2214.157|4250;18000;2210.122" 
    lgl_infs = [4, 4.5, 5, 6] #preparing the k infinity values
    csts = copy.deepcopy(cst.CONSTS) #not to touch the actual constants dict
    load_case = cst.LOAD_C[2] #landing load case used in the sensitivity study
    load_case["FT"] = 5000 #landing at full thrust - a weird load case that tests everything at once

    for lgl in lgl_infs:
        csts["LGL"] = lgl
        eleDict = ed.eledict(csts, cst.INITIAL, cst.CODES)
        meshOut = mesh_block(data, cst.INITIAL, eleDict, csts, cst.CODES)
        sol = fem_linear_block(csts, meshOut, load_case, True, True)
        wfig = plot_block(sol['w'], "w", meshOut, csts)
        wfig.savefig(fr"C:\marek\studia\hpb\Results\Sensitivity Study LGL\w\K{lgl}.pdf")
        vfig = plot_block(sol['v'], "v", meshOut, csts)
        vfig.savefig(fr"C:\marek\studia\hpb\Results\Sensitivity Study LGL\v\K{lgl}.pdf")
        print(f"{lgl} is done")