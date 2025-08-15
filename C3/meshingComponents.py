import geometricClasses as gcl
import typing as ty
import numpy.typing as nt
import numpy as np
import ptsFromCAD as pfc

'''Here be a file with meshings of given components. 
Inside-component connections will be completed in this file, 
some connections might have to be applied in main'''

'''First a modification of multi section sheet from gcl, that allows for irregular rib spacing'''
def ribbed_sheet_y(pt1s:ty.List[gcl.Point3D], pt2s:ty.List[gcl.Point3D], ys:ty.List[float], nby:int, nLs:ty.List[int], returnSecIds=False):
    #Requirements, same y for all pt1s and all pt2s, matching ys[0] and ys[-1], respectively.
    #This is satisfied by the data from pfc
    extpt1s = [pt1s[0]]
    extpt2s = [pt2s[0]]
    for nL, p11, p12, p21, p22 in zip(nLs, pt1s[:-1], pt1s[1:], pt2s[:-1], pt2s[1:]):
        extpt1s += p11.pts_between(p12, nL)[1:]
        extpt2s += p21.pts_between(p22, nL)[1:]

    extys = [ys[0]]
    if nby is None: #for this mode ys have to be ascending!!!
        assert all(y1<y2 for y1, y2 in zip(ys[:-1], ys[1:]))
        effective_nL = sum(nLs)+2-len(nLs)
        x1, _, _ = gcl.pts2coords3D(pt1s)
        x2, _, _ = gcl.pts2coords3D(pt2s)
        deltax1 = max(x1)-min(x1)
        deltax2 = max(x2)-min(x1)
        delta_x = lambda y:deltax1*(1-y/ys[-1])+y/ys[-1]*deltax2
        nbylocs = [1]

    for y1, y2 in zip(ys[:-1], ys[1:]):
        if nby is None:
            nbyloc = int(np.ceil((y2-y1)*effective_nL/delta_x(y2))) #dynamically adjusting nb to suit our aspect ratio needs
            extys += list(np.linspace(y1, y2, nbyloc)[1:])
            nbylocs.append(nbyloc)
        else:
            extys += list(np.linspace(y1, y2, nby)[1:])

    lines = [gcl.Line3D.from_pts(pt1, pt2) for pt1, pt2 in zip(extpt1s, extpt2s)]
    pts = np.array([[line.for_y(y) for line in lines] for y in extys])

    #copy-paste from gcl
    if returnSecIds: #returning the indexes of the sections if asked for
        secIndexes = [0] #section index is not id!!!
        secIdx = 0 #section index is the commond value of array second dimension for the section on the sheet
        for nL in nLs:
            secIdx+=nL-1
            secIndexes.append(secIdx)
        assert secIdx == pts.shape[1]-1
        if nby is None:
            idxbs = np.cumsum(np.array(nbylocs)-1) #getting the indices along span-also important as we also have irregular spacings there
            return pts, secIndexes, idxbs
        return pts, secIndexes, idxbs
    
    return pts

def skin(mesh:gcl.Mesh3D, up:pfc.UnpackedPoints, ys:ty.List[float], skinCode:str, ncCoeff=1., nbuckl=5):
    nc = int(np.ceil(ncCoeff*nbuckl)) #allowing for possible refinements
    def top_or_bot_sh(pt1s:ty.List[gcl.Point3D], pt2s:ty.List[gcl.Point3D], surfaceFun:ty.Callable):
        nLs = [nc]*(len(pt1s)-1)
        rss, idxs, idys = ribbed_sheet_y(pt1s, pt2s, ys, None, nLs, True)
        sss = gcl.project_np3D(rss, zupdate=lambda x, y, z:surfaceFun(x, y))
        ids = mesh.register(sss.flatten().tolist())
        idg = ids.reshape(sss.shape)
        mesh.quad_interconnect(idg, skinCode)
        return idg, sss, idxs, idys
    
    idgt, ssst, idxt, idyt = top_or_bot_sh(up.fcps[1::2], up.tcps[1::2], up.surft)
    idgb, sssb, idxb, idyb = top_or_bot_sh(up.fcps[::2], up.tcps[::2], up.surfb) 
    assert np.allclose(idxt, idxb), f"idxt: {idxt}; idxb: {idxb}"
    assert np.allclose(idyt, idyb), f"idxt: {idyt}; idxb: {idyb}"
    return idgt, ssst, idgb, sssb, idxb, idyb

def truss(mesh:gcl.Mesh3D, up:pfc.UnpackedPoints, lgendidx:int, idgt:nt.NDArray[np.int32], ssst:nt.ArrayLike, 
          idgb:nt.NDArray[np.int32], sssb:nt.ArrayLike, idxs:ty.List[int], idys:ty.List[int],
          lgt:str, LETEt:str, spart:str, rail:str, ribt:str, mbattot:float):
    "1) extending the list of spanwise indices to truss locations"
    vertical_truss_indices = [idys[0]]
    for id1, id2 in zip(idys[:-1], idys[1:]):
        n_extra = round(max(0, 2*(ssst[id2,0].y-ssst[id1,0].y)/(ssst[id1,0].z-sssb[id1,0].z+ssst[id2,0].z-sssb[id2,0].z))-1)
        if n_extra:
            vertical_truss_indices += [round(index) for index in np.linspace(id1, id2, n_extra+2)[1:-1]]
        vertical_truss_indices.append(id2)

    "2) mass proportionalities"
    #proportionality will be based on ~c^2 and A^2
    #chord proportionality
    c2s = list() 
    for idx in range(ssst.shape[0]):
        c2s.append(up.c_at_y(ssst[idx, 0].y)**2) #all trusses at a spanwise idx should have same spanwise coord
    sum_c2s = sum(c2s)
    c2_fracs = [c2/sum_c2s for c2 in c2s] #chordwise fractions
    
    As = list()
    for p1, pidx, p2 in zip(up.fcps[1:-3], up.fcps[2:-2], up.fcps[3:-1]):
        #triangle area computation
        As.append(.5*np.linalg.norm(np.cross([p1.x-pidx.x, p1.y-pidx.y, p1.z-pidx.z], [p2.x-pidx.x, p2.y-pidx.y, p2.z-pidx.z])))
    sumAs = sum(As)
    As_fracs = [area/sumAs for area in As]

    #NOTE: design dependent!
    Afts = As_fracs[1::2]
    Afbs = As_fracs[::2]

    "3) rails-regular"
    for idx, Aft, Afb in zip(idxs[1:-1], Afts, Afbs):
        mesh.beam_interconnect(idgt[:, idx], rail)
        mesh.beam_interconnect(idgb[:, idx], rail)
        for bidx, cf in enumerate(c2_fracs):
            #rough cg of the bettery estimate
            cgt = gcl.Direction3D(0,0,-1).step(ssst[bidx, idx], 2/3*(ssst[bidx, idx].z-sssb[bidx, idx].z))
            mesh.inertia_attach(Aft*cf*mbattot, idgt[bidx, idx], cgt)
            cgb = gcl.Direction3D(0,0,1).step(sssb[bidx, idx], 2/3*(ssst[bidx, idx].z-sssb[bidx, idx].z))
            mesh.inertia_attach(Afb*cf*mbattot, idgb[bidx, idx], cgb)

    "4) ribs"
    for idxnum, idx in enumerate(idys):
        if idxnum<=lgendidx:
            mesh.beam_interconnect(idgt[idx, :idxs[-2]+1], ribt)
            mesh.beam_interconnect(idgb[idx, :idxs[-2]+1], ribt)
            #reinforced landing gear ribs
            mesh.beam_interconnect(idgt[idx, idxs[-2]:], lgt)
            mesh.beam_interconnect(idgb[idx, idxs[-2]:], lgt)
        else:
            mesh.beam_interconnect(idgt[idx, :], ribt)
            mesh.beam_interconnect(idgb[idx, :], ribt)

    "5) special booms"
    mesh.beam_interconnect(idgt[:, 0], LETEt)
    mesh.beam_interconnect(idgt[idys[lgendidx]:, -1], LETEt)
    mesh.beam_interconnect(idgt[:idys[lgendidx]+1, -1], lgt)
    mesh.beam_interconnect(idgb[:, 0], LETEt)
    mesh.beam_interconnect(idgb[idys[lgendidx]:, -1], LETEt)
    mesh.beam_interconnect(idgb[:idys[lgendidx]+1, -1], lgt)

    "6) verticals"
    for truss_idx_b in vertical_truss_indices:
        mesh.beam_connect([idgt[truss_idx_b, 0]], [idgb[truss_idx_b, 0]], LETEt)
        if truss_idx_b<=idys[lgendidx]:
            mesh.beam_connect([idgt[truss_idx_b, -1]], [idgb[truss_idx_b, -1]], lgt)
            mesh.beam_connect([idgt[truss_idx_b, idxs[-2]]], [idgb[truss_idx_b, -1]], lgt)
        else:
            mesh.beam_connect([idgt[truss_idx_b, -1]], [idgb[truss_idx_b, -1]], LETEt)
            mesh.beam_connect([idgt[truss_idx_b, idxs[-2]]], [idgb[truss_idx_b, -1]], LETEt)
        for idxc, topidxconn in zip(idxs[1:-1], idxs[:-2]):
            mesh.beam_connect([idgt[truss_idx_b, idxc]], [idgb[truss_idx_b, idxc]], spart)
            mesh.beam_connect([idgt[truss_idx_b, topidxconn]], [idgb[truss_idx_b, idxc]], spart)

    "7) diagonals - this way to put diagonals in tension"
    for truss_idx_b1, truss_idx_b2 in zip(vertical_truss_indices[:-1], vertical_truss_indices[1:]):
        mesh.beam_connect([idgt[truss_idx_b1, 0]], [idgb[truss_idx_b2, 0]], LETEt)
        if truss_idx_b2<=idys[lgendidx]:
            mesh.beam_connect([idgt[truss_idx_b1, -1]], [idgb[truss_idx_b2, -1]], lgt)
            mesh.beam_connect([idgt[truss_idx_b1, idxs[-2]]], [idgb[truss_idx_b2, -1]], lgt)
        else:
            mesh.beam_connect([idgt[truss_idx_b1, -1]], [idgb[truss_idx_b2, -1]], LETEt)
            mesh.beam_connect([idgt[truss_idx_b1, idxs[-2]]], [idgb[truss_idx_b2, -1]], LETEt)
        for idxc, topidxconn in zip(idxs[1:-1], idxs[:-2]):
            mesh.beam_connect([idgt[truss_idx_b1, idxc]], [idgb[truss_idx_b2, idxc]], spart)
            mesh.beam_connect([idgt[truss_idx_b1, topidxconn]], [idgb[truss_idx_b2, idxc]], spart)  

def motors(mesh:gcl.Mesh3D, motors:ty.List[gcl.Point3D], motorR:float, motorL:float, motormass:float):
    sum_in_mn = sum([ine.mn for ine in mesh.inertia])
    for mtr in motors:
        affected_ids = list()
        for i, node in enumerate(mesh.nodes):
            if (node.y >= mtr.y-motorR) and (node.y <= motorR+mtr.y) and (node.x-mtr.x <= motorL/2):
                affected_ids.append(i)
        mn = motormass/len(affected_ids)
        for i in affected_ids:
            mesh.inertia_attach(mn, i, mtr)
    assert np.isclose(sum([ine.mn for ine in mesh.inertia])-sum_in_mn, 4*motormass)

def lg(mesh:gcl.Mesh3D, uplg:gcl.Point3D, lgM, lgR, lgL, lgendidx:int, idgt:nt.NDArray[np.int32], idgb:nt.NDArray[np.int32],
       ribyidxs:ty.List[int], idxs:ty.List[int]):
    sum_in_mn = sum([ine.mn for ine in mesh.inertia])
    lgidxs = list(idgt[ribyidxs[lgendidx-2]:ribyidxs[lgendidx]+1, [idxs[-2], idxs[-1]]].flatten())
    lgidxs += list(idgb[ribyidxs[lgendidx-2]:ribyidxs[lgendidx]+1, [idxs[-2], idxs[-1]]].flatten())
    [mesh.inertia_attach(lgM/len(lgidxs), idx, uplg) for idx in lgidxs]
    assert np.isclose(sum([ine.mn for ine in mesh.inertia])-sum_in_mn, lgM)
    return lgidxs


def hinge(mesh:gcl.Mesh3D, uphinge:gcl.Point3D, panidTop:nt.NDArray[np.int64], mhn:float):
    sum_in_mn = sum([ine.mn for ine in mesh.inertia])
    xids = panidTop[-1, :]
    mn = mhn/len(xids)
    [mesh.inertia_attach(mn, xid, uphinge) for xid in xids]
    assert np.isclose(sum([ine.mn for ine in mesh.inertia])-sum_in_mn, mhn)


def LETE(mesh:gcl.Mesh3D, lineLE:gcl.Line3D, lineTE:gcl.Line3D, panshTop:nt.ArrayLike, panidTop:nt.NDArray[np.int64], panshBot:nt.ArrayLike, panidBot:nt.NDArray[np.int64],
         totmassLE:float, totmassTE:float):
    sum_in_mn = sum([ine.mn for ine in mesh.inertia])
    '''Leading Edge'''
    #connPtsLE = [0]*panshTop.shape[0]
    # connPtsLE[::2] = panshTop[::2, 0]
    # connPtsLE[1::2] = panshBot[1::2, 0]
    connPtsLE = panshBot[::, 0]
    # connIdsLE = [0]*panidTop.shape[0]
    # connIdsLE[::2] = panidTop[::2, 0]
    # connIdsLE[1::2] = panidBot[1::2, 0]
    connIdsLE = panidBot[::, 0]
    connIdsTLE = panidTop[::,0]
    ptsLE = [lineLE.for_y(pt.y) for pt in connPtsLE]
    #the distance between the points is also expressed in terms of chord fractions, it can be used as proprtionality for mass if squared
    distsLE2 = [(pLE.pythagoras(pcLE))**2 for pLE, pcLE in zip(ptsLE, panshBot[:, 0])] #you have to take distance from the same point though
    totDistsLE2 = sum(distsLE2)
    msLE = [d2/totDistsLE2*totmassLE for d2 in distsLE2]
    [mesh.inertia_attach(m/2, id_, pt) for id_, m, pt in zip(connIdsLE, msLE, ptsLE)]
    [mesh.inertia_attach(m/2, id_, pt) for id_, m, pt in zip(connIdsTLE, msLE, ptsLE)]
    '''Trailing Edge'''
    # connPtsTE = [0]*panshTop.shape[0]
    # connPtsTE[::2] = panshTop[::2, -1]
    # connPtsTE[1::2] = panshBot[1::2, -1]
    # connIdsTE = [0]*panidTop.shape[0]
    # connIdsTE[::2] = panidTop[::2, 0-1]
    # connIdsTE[1::2] = panidBot[1::2, -1]
    connPtsTE = panshBot[::, -1]
    connIdsTE = panidBot[::, -1]
    connIdsTTE = panidTop[::, -1]
    ptsTE = [lineTE.for_y(pt.y) for pt in connPtsTE]
    #the distance between the points is also expressed in terms of chord fractions, it can be used as proprtionality for mass if squared
    distsTE2 = [(pTE.pythagoras(pcTE))**2 for pTE, pcTE in zip(ptsTE,panshBot[:, -1])]
    totDistsTE2 = sum(distsTE2)
    msTE = [d2/totDistsTE2*totmassTE for d2 in distsTE2]
    [mesh.inertia_attach(m/2, id_, pt) for id_, m, pt in zip(connIdsTE, msTE, ptsTE)]
    [mesh.inertia_attach(m/2, id_, pt) for id_, m, pt in zip(connIdsTTE, msTE, ptsTE)]
    assert np.isclose(sum([ine.mn for ine in mesh.inertia])-sum_in_mn, totmassLE+totmassTE)

#duplicating the ribs code from chat  
def tangle_with_midpoints(lst):
    result = []
    for i in range(len(lst) - 1):
        result.append(lst[i])
        result.append((lst[i] + lst[i + 1]) / 2)
    result.append(lst[-1])  # Add the last point
    return result

def all_components(mesh:gcl.Mesh3D, up:pfc.UnpackedPoints, totmassBat:float, motorR:float, motorL:float, motormass:float, 
                   lgM:float, lgR:float, lgL:float, mhn:float, totmassLE:float, totmassTE:float,
                   skinCode:str, lgCode:str, LETECode:str, sparCode:str, railCode:str, ribCode:str,
                   ncCoeff=1., nbuckl=5):

    '''RIB PLACEMENT DEFINITIONS'''
    tol = .001 #so that the landing gear ribs are captured in the inertia
    ribys = [up.fft.y, (up.fft.y+up.motors[0].y)/2, up.motors[0].y, up.lg.y+tol-lgR, up.lg.y-tol+lgR, up.motors[1].y]
    ribys += [(up.motors[1].y+up.motors[2].y)/2, up.motors[2].y, (up.motors[2].y+up.motors[3].y)/2, up.motors[3].y, up.tft.y]
    ribys = tangle_with_midpoints(ribys) #NOTE:for now i don't think necessary
    lgendid = 8 #used to be 4, but now we added 4 more ribs in between

    #main geometry
    idgt, ssst, idgb, sssb, idxs, idys = skin(mesh, up, ribys, skinCode, ncCoeff, nbuckl)
    truss(mesh, up, lgendid, idgt, ssst, idgb, sssb, idxs, idys, lgCode, LETECode, sparCode, railCode, ribCode, totmassBat)

    #inertia additions
    motors(mesh, up.motors, motorR, motorL, motormass)
    lgids = lg(mesh, up.lg, lgM, lgR, lgL, lgendid, idgt, idgb, idys, idxs)
    hinge(mesh, up.hinge, idgt, mhn)
    LETE(mesh, up.leline, up.teline, ssst, idgt, sssb, idgb, totmassLE, totmassTE)

    return {"skinTop":ssst, "skinBot":sssb}, {"skinTop":idgt, 
            "skinBot":idgb,"lg":lgids}

if __name__ == "__main__":
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    # #test ribbed sheet
    # pt1s = [gcl.Point3D(0, 0, 0), gcl.Point3D(3, 0, .5), gcl.Point3D(4, 0, -.5), gcl.Point3D(4, 0, 4)]
    # pt2s = [gcl.Point3D(0, 6, 1), gcl.Point3D(2.5, 6, 0), gcl.Point3D(3, 6, -.5), gcl.Point3D(4, 6, 5)]
    # ns = 7
    # nLs = [2, 13, 7]
    # msp, idxs = ribbed_sheet_y(pt1s, pt2s, [0, 1, 4, 4.5, 6], None, nLs, True)

    # plt.figure()
    # ax = plt.axes(projection="3d")
    # ax.scatter(*gcl.pts2coords3D(msp.flatten()))

    import ptsFromCAD as pfc
    data = "5000;0;-6.5|4750;0;-24|4500;0;-41|4000;0;-75|3500;0;-107|3000;0;-138|2500;0;-167|2000;0;-190|1500;0;-206|1250;0;-211|1000;0;-211.5|750;0;-205|500;0;-187.5|375;0;-173|250;0;-150.5|125;0;-113.5|62.5;0;-82.5|0;0;0|62.5;0;107.5|125;0;149.5|250;0;206.5|375;0;248|500;0;281.5|750;0;330.5|1000;0;363|1250;0;383.5|1500;0;394|2000;0;390|2500;0;362|3000;0;318|3500;0;259|4000;0;187.5|4500;0;104|4750;0;57|5000;0;6.5&4250;18000;2206.872|4125;18000;2198.122|4000;18000;2189.622|3750;18000;2172.622|3500;18000;2156.622|3250;18000;2141.122|3000;18000;2126.622|2750;18000;2115.122|2500;18000;2107.122|2375;18000;2104.622|2250;18000;2104.372|2125;18000;2107.622|2000;18000;2116.372|1937.5;18000;2123.622|1875;18000;2134.872|1812.5;18000;2153.372|1781.25;18000;2168.872|1750;18000;2210.122|1781.25;18000;2263.872|1812.5;18000;2284.872|1875;18000;2313.372|1937.5;18000;2334.122|2000;18000;2350.872|2125;18000;2375.372|2250;18000;2391.622|2375;18000;2401.872|2500;18000;2407.122|2750;18000;2405.122|3000;18000;2391.122|3250;18000;2369.122|3500;18000;2339.622|3750;18000;2303.872|4000;18000;2262.122|4125;18000;2238.622|4250;18000;2213.372&-471.576;3673.46;441.249|126.713;7770.33;945.522|725.002;11867.2;1449.796|1323.292;15964.069;1954.07&3012.266;18950.167;2324.854|4861.588;5721.895;702.56&500;0;0|1999.77;18000;2214.157&3750;0;0|3624.77;18000;2214.157&871.849;1600;0.583|871.849;1600;512.208|1171.071;1600;-6.06|1503.866;1600;570.358|1829.363;1600;6.581|2152.161;1600;565.683|2460.005;1600;32.482|2745.201;1600;526.456|3012.213;1600;63.976|3249.622;1600;475.18|3470.631;1600;92.38|3470.631;1600;447.784&2124.805;18000;2107.631|2124.805;18000;2375.342|2281.374;18000;2104.155|2455.511;18000;2405.769|2625.829;18000;2110.769|2794.735;18000;2403.323|2955.816;18000;2124.322|3105.047;18000;2382.797|3244.763;18000;2140.802|3368.988;18000;2355.966|3484.632;18000;2155.664|3484.632;18000;2341.631&0;0;0|5000;0;0|1749.77;18000;2214.157|4250;18000;2210.122" 
    up = pfc.UnpackedPoints(data)
    mesh = gcl.Mesh3D()
    #nb = 50
    na = 7
    nf2 = 3
    ntrig = 2
    dz = .015
    din = .010
    #nip = 18
    cspacing = .25
    bspacing = 1
    totmass = 17480
    lemass = 1000
    temass = 3000
    lgR=.5
                                  
    all_components(mesh, up, totmass, .5, 3.5, 1000, 5000, 1, 4.5, 1000, lemass, temass, "sk", "lg", "lt", "sp", "rl", "rb")
    

    plt.figure()
    #comparison of what is registered in the mesh and what the sheets are
    ax = plt.axes(projection="3d")
    specialpts = []
    ax.scatter(*gcl.pts2coords3D(specialpts))
    mesh.visualise(ax)
    ax.legend()
    plt.show()


