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
    for y1, y2 in zip(ys[:-1], ys[1:]):
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
        return pts, secIndexes
    
    return pts



def trigspars(mesh:gcl.Mesh3D, ys:ty.List[float], nb:int, na:int, nf2:int, ntrig:int, 
              shell:str, ffb:gcl.Point3D, frb:gcl.Point3D, 
              frt:gcl.Point3D, fft:gcl.Point3D,tfb:gcl.Point3D, 
              trb:gcl.Point3D, trt:gcl.Point3D, tft:gcl.Point3D
              )->ty.Tuple[ty.List[nt.ArrayLike]]:
    def trig_crossec(fb:gcl.Point3D, rb:gcl.Point3D, rt:gcl.Point3D, 
                     ft:gcl.Point3D)->ty.List[gcl.Point3D]:
        a = fb.pythagoras(ft) #the side of the battery triangle
        f = (ft.pythagoras(rt)-a*ntrig)/(2*ntrig+1) #the full flange width
        f2 = f/2 #half flange wit=dth used for until-rivet modelling
        dirc = gcl.Direction3D(1, 0, 0)
        diru = gcl.Direction3D(1, 0, np.sqrt(3))
        dird = gcl.Direction3D(1, 0, -np.sqrt(3))
        
        #1) first sheet, the inwards curved
        inf = dirc.step(fb, f2) #inwards flange midpoint
        tfp = dirc.step(ft, f2) #top flange midpoint - updated later
        pts = [inf, fb, ft, tfp] #output - list of lists of points

        #2) creating all the middle sheets
        for i in range(ntrig):
            #2.1) downhill sheet
            cpt = dirc.step(tfp, f2) #corner point top
            cpb = dird.step(cpt, a) #corner point bottom
            bfp = dirc.step(cpb, f2) #bottom flange midpoint
            pts+=[cpt, cpb, bfp]
            #2.2) uphill sheet
            cpb = dirc.step(bfp, f2) #the pts get updated
            cpt = diru.step(cpb, a)
            tfp2 = dirc.step(cpt, f2) #update with check
            assert np.isclose(tfp2.x, tfp.x+a+f*2)
            tfp = tfp2
            pts+=[cpb, cpt, tfp]
        
        #3) creating the final sheet
        assert np.isclose(tfp.x, rt.x-f2)
        #again, an inwards-facing flange
        pts+=[rt, rb, dirc.step(rb, -f2)]

        return pts, a, f
    
    #appling the cross section generation to get sheets
    pt1s, a1, f = trig_crossec(ffb, frb, frt, fft)
    pt2s, a2, _ = trig_crossec(tfb, trb, trt, tft)

    sheet, secIdxs = ribbed_sheet_y(pt1s, pt2s, ys, nb, [nf2, na, nf2]*(2*ntrig+2), True)
    idspace = mesh.register(list(sheet.flatten()))
    idgrid = idspace.reshape(sheet.shape)
    #interconnection of sheets
    mesh.quad_interconnect(idgrid, shell)

    return sheet, idgrid, secIdxs, a1, a2, f


def bat_rail(mesh:gcl.Mesh3D, ntrig:int, a1:float, a2:float, f:float, 
             dz:float, midflids:ty.List[int], rail:str,totmass:float)->ty.Tuple[ty.List[int]]:
    '''nodes for midflids have to be oriented from y=0 to y=ymax'''
    zaxis = gcl.Direction3D(0,0,1)
    mesh.beam_interconnect(midflids, rail)

    #batteries - harder job. We need to define the local battery area
    #continuous attachment for now - subject to change
    props = dict()
    dzs = dict()
    for i in midflids:
        y = mesh.nodes[i].y
        y0 = mesh.nodes[midflids[0]].y
        ymax = mesh.nodes[midflids[-1]].y
        yfrac = (y-y0)/(ymax-y0)
        a = a2*yfrac+(1-yfrac)*a1 #linear interpolation of battery a
        #battery volume proportionality - july 10th page TODO:SUS
        prop = (a+2*f)*a*np.sqrt(3)/4
        assert prop>0 #if prop<0, then we have a prob - foil too thin at the tip
        props[i] = prop
        dzs[i] = np.sign(dz)*a*np.sqrt(3)/3 #oriented distance from the rail to the battery centroid
    propmass = totmass/(4*ntrig+2)/sum(props.values()) #how much mass a unit prop means
    batids = list()
    for i in midflids:
        batpt = zaxis.step(mesh.nodes[i], dzs[i])
        #assigning the variable mass to the battery using protocol
        mesh.inertia_attach(props[i]*propmass, i, batpt)

    return batids

def panel(mesh:gcl.Mesh3D, ff:gcl.Point3D, fr:gcl.Point3D, tf:gcl.Point3D, tr:gcl.Point3D, nb:int, ribys:ty.List[float], nip:int, nf:int, 
          floor:str, skin:str, rib:str, flange:str, panelz:ty.Callable, ribconnids:ty.List[nt.NDArray[np.int64]],
          cspacing:float)->ty.Tuple[ty.List[int]]:
    #1) lower sheet
    ribconnpt1s = [mesh.nodes[ptids[0]] for ptids in ribconnids]
    ribconnpt2s = [mesh.nodes[ptids[-1]] for ptids in ribconnids]
    floorsh, secs = ribbed_sheet_y([ff]+ribconnpt1s+[fr], [tf]+ribconnpt2s+[tr], ribys, nb,
                                              [nf]+[nip]*(len(ribconnpt1s)-1)+[nf], True)
    #1.1) sheet registration without the already registered connection points
    ribconnIdsTospend = ribconnids.copy()
    flidgrid = np.zeros(floorsh.shape, np.int32)
    for i in range(floorsh.shape[1]):
        if i in secs[1:-1]: #we have to replace with the original connection points
            flidgrid[:, i] = ribconnIdsTospend.pop(0)
            floorsh[:, i] = [mesh.nodes[id_] for id_ in flidgrid[:, i]]
        else:
            flidgrid[:, i] = mesh.register(list(floorsh[:, i]))
    
    mesh.quad_interconnect(flidgrid, floor)

    #2) upper sheet
    upsh = gcl.project_np3D(floorsh, zupdate=lambda x,y,z:panelz(x,y))
    upidspace = mesh.register(list(upsh.flatten()))
    upidgrid = upidspace.reshape(upsh.shape)
    mesh.quad_interconnect(upidgrid, skin)    

    #3) obligatory stiffeners
    def stiffener(pt1s, pt2s, dir):
        mesh.quad_connect(pt1s, pt2s, rib)
        mesh.beam_interconnect(pt1s, flange, dir)
        mesh.beam_interconnect(pt2s,flange, dir)

    [stiffener(upidgrid[:, sec], flidgrid[:, sec], (1, 0, 0)) for sec in secs]


    #4) chordwise stiffeners
    #we use bspacing - spacing along the span, as chordwise stiffeners are
    ribNbs = range(0, flidgrid.shape[0], nb-1) #plus 1 as you have to close the panel
    [stiffener(upidgrid[ribNb, :], flidgrid[ribNb, :], (0, tf.y-ff.y, tf.z-ff.z)) for ribNb in ribNbs] #creating the stiffeners
    
    #5) extra spanwise stiffeners
    for ribNb1, ribNb2 in zip(ribNbs[:-1], ribNbs[1:]):
        for sec1, sec2 in zip(secs[:-1], secs[1:]):
            ccount = int(np.ceil((mesh.nodes[flidgrid[ribNb1, sec2]].x-mesh.nodes[flidgrid[ribNb1, sec1]].x)/cspacing))
            assert nip/(ccount) >= 4
            if ccount>1:
                ribNcs = [int(np.rint(i)) for i in np.linspace(sec1, sec2, ccount+1)[1:-1]]
                [stiffener(upidgrid[ribNb1:(ribNb2+1), ribNc], flidgrid[ribNb1:(ribNb2+1), ribNc], (1, 0, 0)) for ribNc in ribNcs]

    return floorsh, upsh, flidgrid, upidgrid


def motors(mesh:gcl.Mesh3D, motors:ty.List[gcl.Point3D], motorR:float, motorL:float, motormass:float):
    for mtr in motors:
        affected_ids = list()
        for i, node in enumerate(mesh.nodes):
            if (node.y >= mtr.y-motorR) and (node.y <= motorR+mtr.y) and (node.x-mtr.x <= motorL/2):
                affected_ids.append(i)
        mn = motormass/len(affected_ids)
        for i in affected_ids:
            mesh.inertia_attach(mn, i, mtr)

def lg(mesh:gcl.Mesh3D, uplg:gcl.Point3D, lgM, lgR, lgL):
    return mesh.inertia_smear(lgM, uplg, gcl.Point3D(uplg.x-lgL/2, uplg.y-lgR, uplg.z-lgR), gcl.Point3D(uplg.x+lgL/2, uplg.y+lgR, uplg.z+lgR))


def hinge(mesh:gcl.Mesh3D, uphinge:gcl.Point3D, panidTop:nt.NDArray[np.int64], mhn:float):
    xids = panidTop[-1, :]
    mn = mhn/len(xids)
    [mesh.inertia_attach(mn, xid, uphinge) for xid in xids]


def LETE(mesh:gcl.Mesh3D, lineLE:gcl.Line3D, lineTE:gcl.Line3D, panshTop:nt.ArrayLike, panidTop:nt.NDArray[np.int64], panshBot:nt.ArrayLike, panidBot:nt.NDArray[np.int64],
         totmassLE:float, totmassTE:float):
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
    


def all_components(mesh:gcl.Mesh3D, up:pfc.UnpackedPoints, nbCoeff:float, na:int, nf2:int, nipCoeff:float, ntrig:int, cspacing:float, 
                   totmass:float, totmassLE:float, totmassTE:float, spar:str, plate:str, rib:str, flange:str, skin:str, rail:str, motorM:float, motorR:float,
                   motorL, lgmass:float, lgR:float, lgL:float, hinge_:float):

    nb = 5*nbCoeff

    '''RIB PLACEMENT DEFINITIONS'''
    tol = .001 #so that the landing gear ribs are captured in the inertia
    ribys = [up.fft.y, (up.fft.y+up.motors[0].y)/2, up.motors[0].y, up.lg.y+tol-lgR, up.lg.y-tol+lgR, up.motors[1].y]
    ribys += [(up.motors[1].y+up.motors[2].y)/2, up.motors[2].y, (up.motors[2].y+up.motors[3].y)/2, up.motors[3].y, up.tft.y]

    sparsh, sparigrd, sparSecIdxs, a1, a2, f = trigspars(mesh, ribys, nb, na, nf2, ntrig, spar, up.ffb, up.frb, up.frt, up.fft, up.tfb, up.trb, up.trt, up.tft)

    nip = int(np.ceil(nipCoeff*(4*((a1+2*f)/cspacing+1)))) #obtaining nip so that we get the minimal required amount of elements

    sparBotConns = sparSecIdxs[0::6]
    sparTopConns = sparSecIdxs[3::6]
    sbci = [sparigrd[:,idx] for idx in sparBotConns] #spar bottom connection ids
    stci = [sparigrd[:,idx] for idx in sparTopConns] #spar top connection ids
    batTopConn1s = sparSecIdxs[2::6]
    batBotConn1s = sparSecIdxs[5::6][:-1] #this one would also have the semi last corner which we know should not be there
    sbbc1 = [sparigrd[:,idx] for idx in batBotConn1s] 
    stbc1 = [sparigrd[:,idx] for idx in batTopConn1s]
    batTopConn2s = sparSecIdxs[4::6]
    batBotConn2s = sparSecIdxs[7::6] #this one would also have the semi last corner which we know should not be there
    sbbc2 = [sparigrd[:,idx] for idx in batBotConn2s] 
    stbc2 = [sparigrd[:,idx] for idx in batTopConn2s] 


    for i, ids1, ids2 in zip(range(2,4*ntrig+2,4), sbbc1, sbbc2): #bottom batteries
        bat_rail(mesh, ntrig, a1, a2, f, 1, ids1,
                                         rail, totmass)
        bat_rail(mesh, ntrig, a1, a2, f, 1, ids2,
                                         rail, totmass)
    for i, ids1, ids2 in zip(range(0,4*ntrig+2,4), stbc1, stbc2): #top batteries
        bat_rail(mesh, ntrig, a1, a2, f, -1, ids1, #cids,
                                         rail, totmass)
        bat_rail(mesh, ntrig, a1, a2, f, -1, ids2, #cids,
                                         rail, totmass)

    topflsh, topsksh, topflids, topskids = panel(mesh, up.fft, up.frt, up.tft, up.trt, nb, ribys, nip, nf2, 
                                    plate, skin, rib, flange, up.surft, stci, cspacing)
    
    botflsh, botsksh, botflids, botskids = panel(mesh, up.ffb, up.frb, up.tfb, up.trb, nb, ribys, nip, nf2, 
                                    plate, skin, rib, flange, up.surfb, sbci, cspacing)
    
    motors(mesh, up.motors, motorR, motorL, motorM)
    lgids = lg(mesh, up.lg, lgmass, lgR, lgL) #saving the lg id to smear the landing load
    hinge(mesh, up.hinge, topflids, hinge_)
    LETE(mesh, up.leline, up.teline, topflsh, topflids, botflsh, botflids, totmassLE, totmassTE)

    return {"spars":sparsh, "plateTop":topflsh, "skinTop":topsksh, "plateBot":botflsh, "skinBot":botsksh}, {"spars":sparigrd, "plateTop":topflids, "skinTop":topskids, 
            "plateBot":botflids,"skinBot":botskids, "sparBends":sparSecIdxs, "lg":lgids}

if __name__ == "__main__":
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    #test ribbed sheet
    # pt1s = [gcl.Point3D(0, 0, 0), gcl.Point3D(3, 0, .5), gcl.Point3D(4, 0, -.5), gcl.Point3D(4, 0, 4)]
    # pt2s = [gcl.Point3D(0, 6, 1), gcl.Point3D(2.5, 6, 0), gcl.Point3D(3, 6, -.5), gcl.Point3D(4, 6, 5)]
    # ns = 7
    # nLs = [2, 13, 7]
    # msp, idxs = ribbed_sheet_y(pt1s, pt2s, [0, 1, 4, 4.5, 6], 5, nLs, True)

    # plt.figure()
    # ax = plt.axes(projection="3d")
    # ax.scatter(*gcl.pts2coords3D(msp.flatten()))

    import ptsFromCAD as pfc
    data = "5000;0;-6.5|4750;0;-24|4500;0;-41|4000;0;-75|3500;0;-107|3000;0;-138|2500;0;-167|2000;0;-190|1500;0;-206|1250;0;-211|1000;0;-211.5|750;0;-205|500;0;-187.5|375;0;-173|250;0;-150.5|125;0;-113.5|62.5;0;-82.5|0;0;0|62.5;0;107.5|125;0;149.5|250;0;206.5|375;0;248|500;0;281.5|750;0;330.5|1000;0;363|1250;0;383.5|1500;0;394|2000;0;390|2500;0;362|3000;0;318|3500;0;259|4000;0;187.5|4500;0;104|4750;0;57|5000;0;6.5&4250;18000;2206.872|4125;18000;2198.122|4000;18000;2189.622|3750;18000;2172.622|3500;18000;2156.622|3250;18000;2141.122|3000;18000;2126.622|2750;18000;2115.122|2500;18000;2107.122|2375;18000;2104.622|2250;18000;2104.372|2125;18000;2107.622|2000;18000;2116.372|1937.5;18000;2123.622|1875;18000;2134.872|1812.5;18000;2153.372|1781.25;18000;2168.872|1750;18000;2210.122|1781.25;18000;2263.872|1812.5;18000;2284.872|1875;18000;2313.372|1937.5;18000;2334.122|2000;18000;2350.872|2125;18000;2375.372|2250;18000;2391.622|2375;18000;2401.872|2500;18000;2407.122|2750;18000;2405.122|3000;18000;2391.122|3250;18000;2369.122|3500;18000;2339.622|3750;18000;2303.872|4000;18000;2262.122|4125;18000;2238.622|4250;18000;2213.372&-471.576;3673.46;441.249|126.713;7770.33;945.522|725.002;11867.2;1449.796|1323.292;15964.069;1954.07&3012.266;18950.167;2324.854|4861.588;5721.895;702.56&500;0;0|1999.77;18000;2214.157&3500;0;0|3499.77;18000;2214.157&787.633;1600;66.192|2883.375;1600;66.192|2617.418;1600;526.843|1053.59;1600;526.843&2024.321;18000;2152.592|3273.646;18000;2152.592|3148.758;18000;2368.903|2149.209;18000;2368.903" 
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
                                  
    all_components(mesh, up, 1, na, nf2, 1, ntrig, cspacing, totmass, lemass, temass, "/EXCLsp", "/EXCLpl", "/EXCLrb", "/EXCLfl", "sk", "/EXCLrl", 
                   1000, 0.4, 3, 3000, .5, 4, 500)

    plt.figure()
    #comparison of what is registered in the mesh and what the sheets are
    ax = plt.axes(projection="3d")
    # for sh in sheets:
    #     ax.scatter(*gcl.pts2coords3D(np.ravel(sh)))
    mesh.visualise(ax)
    ax.legend()
    plt.show()


