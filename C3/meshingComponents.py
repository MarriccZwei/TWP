import geometricClasses as gcl
import typing as ty
import numpy.typing as nt
import numpy as np

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
        yfrac = (y-y0)/(y-ymax)
        a = a2*yfrac+(1-yfrac)*a1 #linear interpolation of battery a
        #battery volume proportionality - july 10th page
        prop = a**2*np.sqrt(3)/4+a*np.sqrt(3)/2*f-(2*abs(dz)+din)**2*np.sqrt(3)/12
        assert prop>0 #if prop<0, then we have a prob - foil too thin at the tip
        props[i] = prop
        dzs = np.sign(dz)*a*np.sqrt(3)/3-dz #oriented distance from the rail to the battery centroid
    propmass = totmass/sum(props.values()) #how much mass a unit prop means
    batids = list()
    for i in beamids:
        batpt = zaxis.step(mesh.nodes[i], dzs[i])
        batid = mesh.register(batpt)
        #assigning the variable mass to the battery using protocol
        mesh.pointmass_attach(batid, bat, props[i]*propmass)
        mesh.spring_connect([i, batid], batmount)
        batids.append(batids)

    return beamids, batids


if __name__ == "__main__":
    import ptsFromCAD as pfc
    data = "5000;0;-6.5|4750;0;-24|4500;0;-41|4000;0;-75|3500;0;-107|3000;0;-138|2500;0;-167|2000;0;-190|1500;0;-206|1250;0;-211|1000;0;-211.5|750;0;-205|500;0;-187.5|375;0;-173|250;0;-150.5|125;0;-113.5|62.5;0;-82.5|0;0;0|62.5;0;107.5|125;0;149.5|250;0;206.5|375;0;248|500;0;281.5|750;0;330.5|1000;0;363|1250;0;383.5|1500;0;394|2000;0;390|2500;0;362|3000;0;318|3500;0;259|4000;0;187.5|4500;0;104|4750;0;57|5000;0;6.5&4250;18000;2206.872|4125;18000;2198.122|4000;18000;2189.622|3750;18000;2172.622|3500;18000;2156.622|3250;18000;2141.122|3000;18000;2126.622|2750;18000;2115.122|2500;18000;2107.122|2375;18000;2104.622|2250;18000;2104.372|2125;18000;2107.622|2000;18000;2116.372|1937.5;18000;2123.622|1875;18000;2134.872|1812.5;18000;2153.372|1781.25;18000;2168.872|1750;18000;2210.122|1781.25;18000;2263.872|1812.5;18000;2284.872|1875;18000;2313.372|1937.5;18000;2334.122|2000;18000;2350.872|2125;18000;2375.372|2250;18000;2391.622|2375;18000;2401.872|2500;18000;2407.122|2750;18000;2405.122|3000;18000;2391.122|3250;18000;2369.122|3500;18000;2339.622|3750;18000;2303.872|4000;18000;2262.122|4125;18000;2238.622|4250;18000;2213.372&-471.576;3673.46;441.249|126.713;7770.33;945.522|725.002;11867.2;1449.796|1323.292;15964.069;1954.07&3012.266;18950.167;2324.854|4861.588;5721.895;702.56&500;0;0|1999.77;18000;2214.157&3500;0;0|3499.77;18000;2214.157&787.633;1600;66.192|2883.375;1600;66.192|2617.418;1600;526.843|1053.59;1600;526.843&2024.321;18000;2152.592|3273.646;18000;2152.592|3148.758;18000;2368.903|2149.209;18000;2368.903" 
    up = pfc.UnpackedPoints(data)
    mesh = gcl.Mesh3D()
    nb = 50
    na = 7
    nf2 = 3
    ntrig = 2
    dz = 15
    din = 10
    sheets, idgrids, _, _, _ = trigspars(mesh, nb, na, nf2, ntrig, "tr", "ts", 
                                up.ffb, up.frb, up.frt, up.fft,
                                up.tfb, up.trb, up.trt, up.tft)
    for igrid in idgrids[::2]:
        edgeids = igrid[-1, :]
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


