import geometricClasses as gcl
import typing as ty
import numpy.typing as nt
import numpy as np
import ptsFromCAD as pfc

'''Here be a file with meshings of given components. 
Inside-component connections will be completed in this file, 
some connections might have to be applied in main'''
def trigspars(mesh:gcl.Mesh3D, nb:int, na:int, nf2:int, ntrig:int, 
              rivet:str, shell:str, ffb:gcl.Point3D, frb:gcl.Point3D, 
              frt:gcl.Point3D, fft:gcl.Point3D,tfb:gcl.Point3D, 
              trb:gcl.Point3D, trt:gcl.Point3D, tft:gcl.Point3D
              )->ty.Tuple[ty.List[nt.ArrayLike]]:
    def trig_crossec(fb:gcl.Point3D, rb:gcl.Point3D, rt:gcl.Point3D, 
                     ft:gcl.Point3D)->ty.List[ty.List[gcl.Point3D]]:
        a = fb.pythagoras(ft) #the side of the battery triangle
        f = (ft.pythagoras(rt)-a*ntrig)/(2*ntrig+1) #the full flange width
        f2 = f/2 #half flange wit=dth used for until-rivet modelling
        dirc = gcl.Direction3D(1, 0, 0)
        diru = gcl.Direction3D(1, 0, np.sqrt(3))
        dird = gcl.Direction3D(1, 0, -np.sqrt(3))
        
        #1) first sheet, the inwards curved
        inf = dirc.step(fb, f2) #inwards flange midpoint
        tfp = dirc.step(ft, f2) #top flange midpoint - updated later
        pts = [[inf, fb, ft, tfp]] #output - list of lists of points

        #2) creating all the middle sheets
        for i in range(ntrig):
            #2.1) downhill sheet
            cpt = dirc.step(tfp, f2) #corner point top
            cpb = dird.step(cpt, a) #corner point bottom
            bfp = dirc.step(cpb, f2) #bottom flange midpoint
            pts.append([tfp, cpt, cpb, bfp])
            #2.2) uphill sheet
            cpb = dirc.step(bfp, f2) #the pts get updated
            cpt = diru.step(cpb, a)
            tfp2 = dirc.step(cpt, f2) #update with check
            assert np.isclose(tfp2.x, tfp.x+a+f*2)
            tfp = tfp2
            pts.append([bfp, cpb, cpt, tfp])
        
        #3) creating the final sheet
        assert np.isclose(tfp.x, rt.x-f2)
        #again, an inwards-facing flange
        pts.append([tfp, rt, rb, dirc.step(rb, -f2)])

        return pts, a, f
    
    #appling the cross section generation to get sheets
    pts1s, a1, f = trig_crossec(ffb, frb, frt, fft)
    pts2s, a2, _ = trig_crossec(tfb, trb, trt, tft)
    sheets = list()
    idgrids = list()
    for pt1s, pt2s in zip(pts1s, pts2s):
        sheet = gcl.multi_section_sheet3D(pt1s, pt2s, nb, [nf2, na, nf2])
        idspace = mesh.register(list(sheet.flatten()))
        idgrid = idspace.reshape(sheet.shape)
        sheets.append(sheet)
        idgrids.append(idgrid)
        #interconnection of sheets
        mesh.quad_interconnect(idgrid, shell)
    
    #riveting sheet boundaries with rivets
    for ids1, ids2 in zip(idgrids[:-1], idgrids[1:]):
        mesh.spring_connect(ids1[:, -1], ids2[:, 0], rivet)

    return sheets, idgrids, a1, a2, f


def bat_rail(mesh:gcl.Mesh3D, ntrig:int, a1:float, a2:float, f:float, 
             din:float, dz:float, midflids:ty.List[int], rail:str, bat:str,
             railmount:str, batmount:str, totmass:float)->ty.Tuple[ty.List[int]]:
    '''nodes for midflids have to be oriented from y=0 to y=ymax'''
    zaxis = gcl.Direction3D(0,0,1)
    #translated points to serve as center of the rails
    transl = [zaxis.step(mesh.nodes[i], dz) for i in midflids] 
    beamids = mesh.register(transl)
    mesh.beam_interconnect(beamids, rail)
    mesh.spring_connect(beamids, midflids, railmount)

    #batteries - harder job. We need to define the local battery area
    #continuous attachment for now - subject to change
    props = dict()
    dzs = dict()
    for i in beamids:
        y = mesh.nodes[i].y
        y0 = mesh.nodes[midflids[0]].y
        ymax = mesh.nodes[midflids[-1]].y
        yfrac = (y-y0)/(ymax-y0)
        a = a2*yfrac+(1-yfrac)*a1 #linear interpolation of battery a
        #battery volume proportionality - july 10th page
        prop = a**2*np.sqrt(3)/4+a*np.sqrt(3)/2*f-(2*abs(dz)+din)**2*np.sqrt(3)/12
        assert prop>0 #if prop<0, then we have a prob - foil too thin at the tip
        props[i] = prop
        dzs[i] = np.sign(dz)*a*np.sqrt(3)/3-dz #oriented distance from the rail to the battery centroid
    propmass = totmass/(2*ntrig+1)/sum(props.values()) #how much mass a unit prop means
    batids = list()
    for i in beamids:
        batpt = zaxis.step(mesh.nodes[i], dzs[i])
        batid = mesh.register([batpt])
        #assigning the variable mass to the battery using protocol
        mesh.pointmass_attach(batid[0], bat, props[i]*propmass)
        mesh.spring_connect([i], batid, batmount)
        batids.append(batids)

    return beamids, batids

def panel(mesh:gcl.Mesh3D, ff:gcl.Point3D, fr:gcl.Point3D, tf:gcl.Point3D, tr:gcl.Point3D, nb:int, nip:int, nf:int, 
          floor:str, skin:str, rib:str, rivet:str, panelz:ty.Callable, ribconnids:ty.List[nt.NDArray[np.int64]],
          cspacing:float, bspacing:float)->ty.Tuple[ty.List[int]]:
    #1) lower sheet
    ribconnpt1s = [mesh.nodes[ptids[0]] for ptids in ribconnids]
    ribconnpt2s = [mesh.nodes[ptids[-1]] for ptids in ribconnids]
    floorsh, secs = gcl.multi_section_sheet3D([ff]+ribconnpt1s+[fr], [tf]+ribconnpt2s+[tr], nb,
                                                 [nf]+[nip]*(len(ribconnpt1s)-1)+[nf], True)
    flidspace = mesh.register(list(floorsh.flatten()))
    flidgrid = flidspace.reshape(floorsh.shape)
    mesh.quad_interconnect(flidgrid, floor)
    for i, ids in enumerate(ribconnids): #riveting the lower sheet to the spars
        panelPts = flidgrid[:, secs[i+1]] #skip the edge subsecs, which are only connected to the top panel
        mesh.spring_connect(panelPts, ids, rivet)

    #2) upper sheet
    upsh = gcl.project_np3D(floorsh, zupdate=lambda x,y,z:panelz(x,y))
    upidspace = mesh.register(list(upsh.flatten()))
    upidgrid = upidspace.reshape(upsh.shape)
    mesh.quad_interconnect(upidgrid, skin)    

    #3) obligatory stiffeners
    mainribidlist = list()
    for sec in secs: #sections are same-number for both sheets due to projection
        #preparing points
        floorids = [flidgrid[i, sec] for i in range(flidgrid.shape[0])]
        upids = [upidgrid[i, sec] for i in range(upidgrid.shape[0])]
        floorpts = floorsh[:, sec]
        upts = upsh[:, sec]

        #stiffener beams
        ribpts = [flp.pt_between(upp) for flp, upp in zip(floorpts, upts)]
        ribids = mesh.register(ribpts)
        #connecting the beam so that a heighte is saved
        for id1, id2, pt2 in zip(ribids[:-1], ribids[1:], floorpts[1:]):
            #taking point 2, so further spanwise, this results in smaller Is, so conservative
            mesh.beam_connect([id1], [id2], rib, abs(panelz(pt2.x, pt2.y)-pt2.z))
        mesh.spring_connect(ribids, floorids, rivet)
        mesh.spring_connect(ribids, upids, rivet)
        mainribidlist.append(ribids)

    #4) chordwise stiffeners
    #we use bspacing - spacing along the span, as chordwise stiffeners are
    bcount = int(np.ceil((tf.y-ff.y)/bspacing)) #more stiffeners chosen if not exact match
    assert nb/bcount >= 4 #elese you have insufficient mesh size and buckling cannot be captured
    ribNbs = [int(np.rint(i)) for i in np.linspace(0, nb-1, bcount+1)] #plus 1 as you have to close the panel
    ribIdbs = []
    for ribNb in ribNbs:
        ribIdbs.append([]) #creating a list for this spanwise position
        #creating extra beams between the obligatory ones
        for sec1, sec2, i in zip(secs[:-1], secs[1:], range(len(mainribidlist)-1)):
            #preparing points, ids and protocols
            floorptsb = floorsh[ribNb, (sec1+1):sec2]
            uptsb = upsh[ribNb, (sec1+1):sec2]
            flooridsb = flidgrid[ribNb, (sec1+1):sec2]
            upidsb = upidgrid[ribNb, (sec1+1):sec2]
            hbs = [abs(upb.z-fpb.z) for upb, fpb in zip(floorptsb, uptsb)]
            #rib creation
            ribptsb = [upb.pt_between(fpb) for upb, fpb in zip(floorptsb, uptsb)]
            ribIdb = mesh.register(ribptsb)
            ribIdbs[-1].append(ribIdb)
            #connections
            mesh.beam_protocolize(ribIdb, rib, hbs[1:]) #chordwise you have to skip one h, you cannot guarantee conservativeness
            mesh.beam_connect([ribIdb[0]], [mainribidlist[i][ribNb]], rib, hbs[0])
            mesh.beam_connect([ribIdb[-1]], [mainribidlist[i+1][ribNb]], rib, hbs[-1])
            mesh.spring_connect(ribIdb, flooridsb, rivet)
            mesh.spring_connect(ribIdb, upidsb, rivet)

    #5) extra spanwise stiffeners - again those use cspacing!
    cribIds = list()
    for ribnear, ribfar, ribNbf, ribNbt in zip(ribIdbs[:-1], ribIdbs[1:], ribNbs[:-1], ribNbs[1:]):
        cribIds.append([])
        for regnear, regfar, sec1, sec2 in zip(ribnear, ribfar, secs[:-1], secs[1:]):
            regdist = mesh.nodes[regnear[-1]].x-mesh.nodes[regnear[0]].x
            cribcount = int(np.ceil(regdist/cspacing))
            cribIds[-1].append([])

            if cribcount>1:
                assert nip/(cribcount+1)>=4 #otherwise mesh refinment insufficient for buckling
                assert regdist > 0 #if we did not loose our coordinate system midway
                #getting indices for brib connections and sheet connections
                #we don't take into account the sides, as they are the obligatory ribs, already created
                idxcs = [int(np.rint(i)) for i in np.linspace(sec1, sec2, cribcount+1)[1:-1]] #indices in c along sheet
                idxregs = [ic-sec1-1 for ic in idxcs] #indices along brib regions
                for idxc, idxr in zip(idxcs, idxregs): #rib creation
                    cribpts = [fpc.pt_between(upc) for fpc, upc in zip(
                        floorsh[(ribNbf+1):(ribNbt), idxc], upsh[ribNbf+1:ribNbt, idxc])]
                    hcs = [abs(fpc.z-upc.z) for fpc, upc in zip(
                        floorsh[(ribNbf+1):(ribNbt), idxc], upsh[ribNbf+1:ribNbt, idxc])]
                    cribids = mesh.register(cribpts)
                    mesh.beam_protocolize(cribids, rib, hcs[1:]) #again heights closer to tip to stay conservative
                    mesh.beam_connect([cribids[0]], [regnear[idxr]], rib, hcs[0])
                    htip = abs(upsh[ribNbt, idxc].z-floorsh[ribNbt, idxc].z) #stay conservative
                    mesh.beam_connect([cribids[-1]], [regfar[idxr]], rib, htip)
                    mesh.spring_connect(cribids, flidgrid[ribNbf+1:ribNbt, idxc], rivet)
                    mesh.spring_connect(cribids, upidgrid[ribNbf+1:ribNbt, idxc], rivet)
                    cribIds[-1].append(cribids)


    return floorsh, upsh, flidgrid, upidgrid, mainribidlist, ribIdbs, cribIds

def motors(mesh:gcl.Mesh3D, motors:ty.List[gcl.Point3D], mountwidth:float, panshTop:nt.ArrayLike, panidTop:nt.NDArray[np.int64], panshBot:nt.ArrayLike, panidBot:nt.NDArray[np.int64],
           motor:str, motormount:str):
    mids = list()
    for mtr in motors:
        #motor generation
        mids+=list(mesh.register([mtr])) #it returns a list, so +=
        mesh.pointmass_attach(mids[-1], motor)
        #boundaries for motormount
        motmount_ymin = mtr.y-mountwidth/2
        motmount_ymax = mtr.y+mountwidth/2
        #connecting the top sheet
        yTop = np.array([pt.y for pt in panshTop[:, 0]])
        checkTop = (yTop>=motmount_ymin) & (yTop<=motmount_ymax)
        ids_2conn_top = panidTop[:, 0][checkTop]
        mesh.spring_connect([mids[-1]]*len(ids_2conn_top), ids_2conn_top, motormount)
        #connecting the bottom sheet
        yBot = np.array([pt.y for pt in panshBot[:, 0]])
        checkBot = (yBot>=motmount_ymin) & (yBot<=motmount_ymax)
        ids_2conn_bot = panidBot[:, 0][checkBot]
        mesh.spring_connect([mids[-1]]*len(ids_2conn_bot), ids_2conn_bot, motormount)
    
    return mids


def all_components(mesh:gcl.Mesh3D, up:pfc.UnpackedPoints, nb:int, na:int, nf2:int, nipCoeff:int, ntrig:int, dzRail:float, dinRail:float, cspacing:float, bspacing:float, totmass:float, 
                   mountwidth:float, rivet:str, spar:str, plate:str, rib:str, skin:str, rail:str, bat:str, batmount:str, railmount:str, motor:str, motormount:str):
    sparsh, sparigrids, a1, a2, f = trigspars(mesh, nb, na, nf2, ntrig, rivet, spar, up.ffb, up.frb, up.frt, up.fft, up.tfb, up.trb, up.trt, up.tft)

    nip = int(np.ceil(nipCoeff*(4*((a1+2*f)/cspacing+1)))) #obtaining nip so that we get the minimal required amount of elements

    beamids, batids = [[]]*len(sparigrids), [[]]*len(sparigrids)
    rivetingids = list()
    for igrid, idx in zip(sparigrids[::2], range(0,len(sparigrids),2)):
        #bats on the upper side go down
        edgeids = igrid[:, -1]
        rivetingids.append(edgeids)
        beamids[idx], batids[idx] = bat_rail(mesh, ntrig, a1, a2, f, dinRail, -dzRail, edgeids, 
                                             rail, bat, railmount, batmount, totmass)
    
    floorsh, upsh, floorids, skinids, ribids, ribIdbs, ribIdcs = panel(mesh, up.fft, up.frt, up.tft, up.trt, nb, nip, nf2, 
                                     plate, skin, rib, rivet, up.surft, rivetingids, cspacing, bspacing)
    
    rivetedbot = [sparigrids[0][:, 0]]
    for igrid, idx in zip(sparigrids[1:-1:2], range(1,len(sparigrids)-1,2)):
        #bats on the upper side go down
        edgeidsN = igrid[:, -1]
        rivetedbot.append(edgeidsN)
        beamids[idx], batids[idx] = bat_rail(mesh, ntrig, a1, a2, f, dinRail, dzRail, edgeidsN, 
                                             rail, bat, railmount, batmount, totmass)
    rivetedbot.append(sparigrids[-1][:, -1])
    fshb, sshb, fib, sib, rib_, rbib, rcib = panel(mesh, up.ffb, up.frb, up.tfb, up.trb, nb, nip, nf2, 
                                      plate, skin, rib, rivet, up.surfb, rivetedbot, cspacing, bspacing)
    
    #motors
    mids = motors(mesh, up.motors, mountwidth, floorsh, floorids, fshb, fib, motor, motormount)

    return {"spars":sparsh, "plateTop":floorsh, "skinTop":upsh, "plateBot":fshb, "skinBot":sshb}, {"spars":sparigrids, "plateTop":floorids, "skinTop":skinids, "plateBot":fib,
         "skinBot":sib, "ribsTop":(ribids, ribIdbs, ribIdcs), "ribsBot":(rib_, rbib, rcib), "rails":beamids, "bats":batids, "motors":mids}

if __name__ == "__main__":
    import ptsFromCAD as pfc
    data = "5000;0;-6.5|4750;0;-24|4500;0;-41|4000;0;-75|3500;0;-107|3000;0;-138|2500;0;-167|2000;0;-190|1500;0;-206|1250;0;-211|1000;0;-211.5|750;0;-205|500;0;-187.5|375;0;-173|250;0;-150.5|125;0;-113.5|62.5;0;-82.5|0;0;0|62.5;0;107.5|125;0;149.5|250;0;206.5|375;0;248|500;0;281.5|750;0;330.5|1000;0;363|1250;0;383.5|1500;0;394|2000;0;390|2500;0;362|3000;0;318|3500;0;259|4000;0;187.5|4500;0;104|4750;0;57|5000;0;6.5&4250;18000;2206.872|4125;18000;2198.122|4000;18000;2189.622|3750;18000;2172.622|3500;18000;2156.622|3250;18000;2141.122|3000;18000;2126.622|2750;18000;2115.122|2500;18000;2107.122|2375;18000;2104.622|2250;18000;2104.372|2125;18000;2107.622|2000;18000;2116.372|1937.5;18000;2123.622|1875;18000;2134.872|1812.5;18000;2153.372|1781.25;18000;2168.872|1750;18000;2210.122|1781.25;18000;2263.872|1812.5;18000;2284.872|1875;18000;2313.372|1937.5;18000;2334.122|2000;18000;2350.872|2125;18000;2375.372|2250;18000;2391.622|2375;18000;2401.872|2500;18000;2407.122|2750;18000;2405.122|3000;18000;2391.122|3250;18000;2369.122|3500;18000;2339.622|3750;18000;2303.872|4000;18000;2262.122|4125;18000;2238.622|4250;18000;2213.372&-471.576;3673.46;441.249|126.713;7770.33;945.522|725.002;11867.2;1449.796|1323.292;15964.069;1954.07&3012.266;18950.167;2324.854|4861.588;5721.895;702.56&500;0;0|1999.77;18000;2214.157&3500;0;0|3499.77;18000;2214.157&787.633;1600;66.192|2883.375;1600;66.192|2617.418;1600;526.843|1053.59;1600;526.843&2024.321;18000;2152.592|3273.646;18000;2152.592|3148.758;18000;2368.903|2149.209;18000;2368.903" 
    up = pfc.UnpackedPoints(data)
    mesh = gcl.Mesh3D()
    nb = 50
    na = 7
    nf2 = 3
    ntrig = 2
    dz = .015
    din = .010
    #nip = 18
    cspacing = .25
    bspacing = 2
    totmass = 17480
    mountwidth = 1
    # sheets, idgrids, a1, a2, f = trigspars(mesh, nb, na, nf2, ntrig, "tr", "ts", 
    #                             up.ffb, up.frb, up.frt, up.fft,
    #                             up.tfb, up.trb, up.trt, up.tft)
    # beamids, batids = [[]]*len(idgrids), [[]]*len(idgrids)
    # rivetingids = list()
    # for igrid, idx in zip(idgrids[::2], range(0,len(idgrids),2)):
    #     #bats on the upper side go down
    #     edgeids = igrid[:, -1]
    #     rivetingids.append(edgeids)
    #     beamids[idx], batids[idx] = bat_rail(mesh, ntrig, a1, a2, f, din, -dz, edgeids, 
    #                                          "rail", "bat", "rm", "bm", 17480)
    
    # floorsh, upsh, floorids, skinids, ribids, ribIdbs, ribIdcs = panel(mesh, up.fft, up.frt, up.tft, up.trt, nb, nip, nf2, 
    #                                  "panfl", "skin", "rib", "tr", up.surft, rivetingids, cspacing, bspacing)
    
    # rivetedbot = [idgrids[0][:, 0]]
    # for igrid, idx in zip(idgrids[1:-1:2], range(1,len(idgrids)-1,2)):
    #     #bats on the upper side go down
    #     edgeidsN = igrid[:, -1]
    #     rivetedbot.append(edgeidsN)
    #     beamids[idx], batids[idx] = bat_rail(mesh, ntrig, a1, a2, f, din, dz, edgeidsN, 
    #                                          "rail", "bat", "rm", "bm", 17480)
    # rivetedbot.append(idgrids[-1][:, -1])
    # fshb, sshb, fib, sib, rib, rbib, rcib = panel(mesh, up.ffb, up.frb, up.tfb, up.trb, nb, nip, nf2, 
    #                                   "panfl", "skin", "rib", "tr", up.surfb, rivetedbot, cspacing, bspacing)
    pts, ids = all_components(mesh, up, nb, na, nf2, 1, ntrig, dz, din, cspacing, bspacing, totmass, mountwidth, "rv", "sp", "pl", "rb", "sk", "rl", "bt", "bm", "rm", "mo", "mm")

    #comparison of what is registered in the mesh and what the sheets are
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    # for sh in sheets:
    #     ax.scatter(*gcl.pts2coords3D(np.ravel(sh)))
    mesh.visualise(ax)
    ax.legend()
    plt.show()


