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
    pts, ids = mc.all_components(mesh, up, consts["BAT_MASS_1WING"], consts["MR"], consts["ML"], consts["M_MOTOR"], consts["M_LG"],
                                 consts["LGR"], consts["LGL"], consts["M_HINGE"], consts["M_LE"], consts["M_TE"], codes["skin"],
                                 codes["lg"], codes["LETE"], codes["spar"], codes["rail"], codes["rib"], consts["N_COEFF"])
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
    ids_s = np.hstack([ids["skinTop"].flatten(), ids["skinBot"][:, 0].flatten()])
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
    weight = p3g.weight(M, (nult+nlg)*G0, N, pf3.DOF, wdir) if ult else p3g.weight(M, n*G0, N, pf3.DOF, wdir)
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

    return {'u':u, 'w':w, 'v':v, 'bu':bu, 'bk':bk, 'KC0uu':KC0uu, 'W':W, 'ncoords_s':ncoords_s, 'ids_s':ids_s, 'KA':KA, 'KAuu':KAuu}

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

    fig = plt.figure(figsize=(10,6))
    levels = np.linspace(w.min(), w.max(), 50) if not plusonly else np.linspace(0, w.max(), 50)

    def contourp(loc:int, selection:nt.NDArray[np.int32]):
        plt.subplot(loc) if loc != 0 else ""
        sf = selection.flatten()
        plt.contourf(x[sf].reshape(selection.shape), y[sf].reshape(selection.shape), w[sf].reshape(selection.shape), levels=levels)
        plt.xlabel("x [m]", loc="right")
        plt.ylabel("y [m]", loc="top")

    contourp(121, ids["skinBot"])
    plt.title("Lower Skin")
    contourp(122, ids["skinTop"])
    plt.title("Upper Skin")
    cbar = plt.colorbar()
    cbar.set_label(f"{wtxt} [{unit}]")
    fig.subplots_adjust(wspace=0.4)
    return fig


#testing blocks and performing a sensitivity study of springs
if __name__ == "__main__":
    import elementDefs as ed
    import constants as cst
    import copy

    data = cst.CAD_DATA
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