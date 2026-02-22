import aerosandbox as asb
import aerosandbox.numpy as np
import pyvista as pv
import scipy.interpolate as sip
import scipy.optimize as op
import matplotlib.pyplot as plt

class ElysianWing():
    def __init__(self, GEOM_SOURCE:dict[str, float | asb.Airfoil], H_per_c_sq:float, maxrays:int=15, debug:bool=False, xperc_fore_safe:float=.1, xperc_rear_safe=.9):
        #0) Overall wing boundaries
        self.H_per_c_sq = H_per_c_sq
        self.yfus = GEOM_SOURCE["yfus"]
        self.yhn = GEOM_SOURCE["yhn"]
        self.ytip = GEOM_SOURCE["ytip"]
        delta_z_hn_fus = GEOM_SOURCE["deltazhn"]
        self._dz_dy = delta_z_hn_fus/(self.yhn-self.yfus)
        delta_x_hn_fus = GEOM_SOURCE["deltaxhn"]
        self._dx_dy = delta_x_hn_fus/(self.yhn-self.yfus)
        self.xtip = self.xle_at_y(self.ytip)
        self.ztip = self.zle_at_y(self.ytip)

        #1) chords
        self.xperc_fore = GEOM_SOURCE["(x/c)_fore"]
        self.xperc_rear = GEOM_SOURCE["(x/c)_rear"] #NOTE: the target rear spar position, it may not be achieved exactly by the angled spars pattern
        self.cr = GEOM_SOURCE["cr"]
        self.cr_reduced = self.cr*(1-H_per_c_sq)
        self.ct = GEOM_SOURCE["ct"]
        self.ct_reduced = self.ct*(1-H_per_c_sq)

        #1) The landing gear
        self.ylg = GEOM_SOURCE["ylg"]
        delta_x_lg_rearsp = GEOM_SOURCE["deltaxlg"] #NOTE: measurement taken wrt the supposed position of the rear spar, so we will use the supposed position
        clg = self.c_at_y(self.ylg)
        xlelg = self.xle_at_y(self.ylg)
        self.xlg = xlelg+clg*self.xperc_rear+delta_x_lg_rearsp
        self.zlg = self.zle_at_y(self.ylg)
        rlg = GEOM_SOURCE["rlg"]
        self.ylgin = self.ylg-rlg
        self.ylgout = self.ylg+rlg

        #2) The motors
        self.ym1 = GEOM_SOURCE["ym1"]
        self.ym2 = GEOM_SOURCE["ym2"]
        self.ym3 = GEOM_SOURCE["ym3"]
        self.ym4 = GEOM_SOURCE["ym4"]
        xlefus = self.xle_at_y(self.yfus)
        self.xm1 = xlefus+GEOM_SOURCE["deltaxm1"]
        self.xm2 = xlefus+GEOM_SOURCE["deltaxm2"]
        self.xm3 = xlefus+GEOM_SOURCE["deltaxm3"]
        self.xm4 = xlefus+GEOM_SOURCE["deltaxm4"]
        self.zm1 = self.zle_at_y(self.ym1)
        self.zm2 = self.zle_at_y(self.ym2)
        self.zm3 = self.zle_at_y(self.ym3)
        self.zm4 = self.zle_at_y(self.ym4)

        #3) The foils for aerodynamics
        self.rootXSec = asb.WingXSec([0, 0, 0], self.cr, 0., GEOM_SOURCE["rootfoil"])
        self.tipXSec = asb.WingXSec([self.xtip, self.ytip, self.ztip], self.ct, 0., GEOM_SOURCE["tipfoil"])
        self._xperc_fore_safe = xperc_fore_safe
        self._xperc_rear_safe = xperc_rear_safe

        #4) The foils for structural centerline
        self.xleroot_reduced = H_per_c_sq/2*self.cr
        self.xletip_reduced = H_per_c_sq/2*self.ct+self.xtip
        self._rootXSec_reduced = asb.WingXSec([self.xleroot_reduced, 0., 0.], self.cr_reduced, 0., GEOM_SOURCE["rootfoil"])
        self._tipXSec_reduced = asb.WingXSec([self.xleroot_reduced, self.ytip, self.ztip], self.ct_reduced, 0., GEOM_SOURCE['tipfoil'])

        #5) Angled spar scaffold
        #->5.1) the angled spars cross sections
        x_as_fus, z_as_fus, up, maxrays = self._angled_spar_reflections(self.yfus, maxrays)
        x_as_hn, z_as_hn, up_tip, maxrays_hn = self._angled_spar_reflections(self.yhn, maxrays)
        assert up_tip == up
        assert maxrays_hn == maxrays
        self.xperc_rear_result_fus = self.xperc_reduced_from_x(x_as_fus[-1], self.yfus)
        self.xperc_rear_result_hn = self.xperc_reduced_from_x(x_as_hn[-1], self.yhn)
        assert self.xperc_rear_result_hn <= self.xperc_rear_result_fus <= self.xperc_rear
        x_scaff_fus = [x_as_fus[0]]+x_as_fus+[x_as_fus[-1]]
        z_scaff_fus = [self.upper_skin_z(self.xperc_fore, self.yfus)]+z_as_fus+[self.upper_skin_z(self.xperc_rear_result_fus, self.yfus) if up else self.lower_skin_z(self.xperc_rear_result_fus, self.yfus)]
        x_scaff_hn = [x_as_hn[0]]+x_as_hn+[x_as_hn[-1]]
        z_scaff_hn = [self.upper_skin_z(self.xperc_fore, self.yhn)]+z_as_hn+[self.upper_skin_z(self.xperc_rear_result_hn, self.yhn) if up else self.lower_skin_z(self.xperc_rear_result_hn, self.yhn)]    

        if debug:
            plt.plot(x_scaff_fus, z_scaff_fus, label="fus")
            plt.plot(x_scaff_hn, z_scaff_hn, label="hn")
            plt.legend()
            plt.show()

        #->5.2) the y locations of ribs and rib types
        yribs_fus_m1, nribs_fus_m1 = self._intermediate_ribs(self.yfus, self.ym1)
        yribs_m1_lg, nribs_m1_lg = self._intermediate_ribs(self.ym1, self.ylgin)
        yribs_lg_lg, nribs_lg_lg = self._intermediate_ribs(self.ylgin, self.ylgout)
        yribs_lg_m2, nribs_lg_m2 = self._intermediate_ribs(self.ylgout, self.ym2)
        yribs_m2_m3, nribs_m2_m3 = self._intermediate_ribs(self.ym2, self.ym3)
        yribs_m3_m4, nribs_m3_m4 = self._intermediate_ribs(self.ym3, self.ym4) 
        yribs_m4_hn, nribs_m4_hn = self._intermediate_ribs(self.ym4, self.yhn)
        self.yribs = [self.yfus]+yribs_fus_m1+[self.ym1]+yribs_m1_lg+[self.ylgin]+yribs_lg_lg+[self.ylgout]+yribs_lg_m2+[self.ym2]+yribs_m2_m3+[self.ym3]+yribs_m3_m4+[self.ym4]+yribs_m4_hn+[self.yhn]
        self.ribcodes = ["bb"]*(1+nribs_fus_m1)+["mb"]+["bb"]*nribs_m1_lg+["lb"]+["bb"]*nribs_lg_lg+["lb"]+["bb"]*nribs_lg_m2+["mb"]+["bb"]*nribs_m2_m3+["mb"]+["bb"]*nribs_m3_m4+["mb"]+["bb"]*(1+nribs_m4_hn)
        if debug:
            print(self.yribs, self.ribcodes)

        #->5.3) The creation of the scaffold proper
        self.segregated_ribs = {'bb':[], 'lb':[], 'mb':[]} #will become a n_(ribs of this type) X n_(scaffold points per y) X 3 list for each element code
        self.scaffold = list() #will become a n_ribs X n_(scaffold points per y) X 3 numpy array
        for y, code in zip(self.yribs, self.ribcodes):
            yfrac = (y-self.yfus)/(self.yhn-self.yfus)
            ycontribs = list()
            for xsf, xsh, zsf, zsh in zip(x_scaff_fus, x_scaff_hn, z_scaff_fus, z_scaff_hn):
                ycontribs.append([(1-yfrac)*xsf+yfrac*xsh, y, (1-yfrac)*zsf+yfrac*zsh])
            self.segregated_ribs[code].append(ycontribs)
            self.scaffold.append(ycontribs)
        self.scaffold = np.array(self.scaffold)
    

    def c_at_y(self, y:float):
        assert 0.<=y<=self.ytip
        yfrac = y/self.ytip
        return self.cr*(1-yfrac)+self.ct*yfrac
    

    def c_reduced_at_y(self, y:float):
        assert 0.<=y<=self.ytip
        yfrac = y/self.ytip
        return self.cr_reduced*(1-yfrac)+self.ct_reduced*yfrac


    def xle_at_y(self, y:float):
        assert 0.<=y<=self.ytip
        return self._dx_dy*y
    

    def xle_reduced_at_y(self, y:float):
        assert 0.<=y<=self.ytip
        return self._dx_dy*y+self.c_reduced_at_y(y)*self.H_per_c_sq/2


    def zle_at_y(self, y:float):
        assert 0.<=y<=self.ytip
        return self._dz_dy*y
    

    def xsec_at_y(self, y:float):
        assert 0.<=y<=self.ytip
        yfrac = y/self.ytip
        c = self.cr*(1-yfrac)+self.ct*yfrac
        ptle = yfrac*np.array([self.xtip, self.ytip, self.ztip])
        coords = yfrac*self.tipXSec.airfoil.coordinates+(1-yfrac)*self.rootXSec.airfoil.coordinates
        foil = asb.Airfoil(f"{self.rootXSec.airfoil.name}->{yfrac}->{self.tipXSec.airfoil.name}", coords)
        return asb.WingXSec(ptle, c, 0., foil)
    

    def _reduced_xsec_at_y(self, y:float):
        assert 0.<=y<=self.ytip
        yfrac = y/self.ytip
        c = self.cr_reduced*(1-yfrac)+self.ct_reduced*yfrac
        ptle = yfrac*self._tipXSec_reduced.xyz_le+(1-yfrac)*self._rootXSec_reduced.xyz_le
        coords = yfrac*self._tipXSec_reduced.airfoil.coordinates+(1-yfrac)*self._rootXSec_reduced.airfoil.coordinates
        foil = asb.Airfoil(f"{self._rootXSec_reduced.airfoil.name}->{yfrac}->{self._tipXSec_reduced.airfoil.name}", coords)
        return asb.WingXSec(ptle, c, 0., foil)
        

    def upper_skin_z(self, xperc:float, y:float):
        assert self._xperc_fore_safe<=xperc<=self._xperc_rear_safe, xperc
        xsec = self._reduced_xsec_at_y(y)
        foilcoords = xsec.airfoil.coordinates
        foilcoords = foilcoords[(foilcoords[:, 1]>=0.) & (foilcoords[:, 0]<=self._xperc_rear_safe) & (foilcoords[:, 0]>=self._xperc_fore_safe), :]
        interp = sip.CubicSpline(np.flip(foilcoords[:, 0]), np.flip(foilcoords[:, 1]))
        return xsec.chord*interp(xperc)+self.zle_at_y(y)
    
    
    def lower_skin_z(self, xperc:float, y:float):
        assert self._xperc_fore_safe<=xperc<=self._xperc_rear_safe, xperc
        xsec = self._reduced_xsec_at_y(y)
        foilcoords = xsec.airfoil.coordinates
        foilcoords = foilcoords[(foilcoords[:, 1]<=0.) & (foilcoords[:, 0]<=self._xperc_rear_safe) & (foilcoords[:, 0]>=self._xperc_fore_safe), :]
        interp = sip.CubicSpline(foilcoords[:, 0], foilcoords[:, 1])
        return xsec.chord*interp(xperc)+self.zle_at_y(y)
    

    def x_from_xperc(self, xperc:float, y:float):
        return self.xle_at_y(y)+self.c_at_y(y)*xperc
    

    def x_from_xperc_reduced(self, xperc:float, y:float):
        return self.xle_reduced_at_y(y)+self.c_reduced_at_y(y)*xperc
    

    def xperc_from_x(self, x:float, y:float):
        return (x-self.xle_at_y(y))/self.c_at_y(y)
        
    
    def xperc_reduced_from_x(self, x:float, y:float):
        return (x-self.xle_reduced_at_y(y))/self.c_reduced_at_y(y)
    

    def _angled_spar_reflections(self, y:float, maxrays:int):
        xstart = self.x_from_xperc_reduced(self.xperc_fore, y)
        zstart = self.lower_skin_z(self.xperc_fore, y)
        as_xs = [xstart]
        as_zs = [zstart]
        up = True
        for i in range(maxrays):
            if up:
                func = lambda x:self.upper_skin_z(self.xperc_reduced_from_x(x, y), y)-zstart-np.sqrt(3)*(x-xstart)
            else:
                func = lambda x:self.lower_skin_z(self.xperc_reduced_from_x(x, y), y)-zstart+np.sqrt(3)*(x-xstart)
            solution = op.fsolve(func, xstart)
            xperc = self.xperc_reduced_from_x(solution[0], y)

            if xperc<self.xperc_rear:
                xstart = solution[0]
                as_xs.append(xstart)
                zstart = self.upper_skin_z(xperc, y) if up else self.lower_skin_z(xperc, y)
                as_zs.append(zstart)
                up = not up
            else:
                maxrays = i #didn't mange to draw the ith ray, but bound is one above as we count from zero, so i
                break

        return as_xs, as_zs, up, maxrays
    

    def _intermediate_ribs(self, y1:float, y2:float):
        nribs = int((y2-y1)//self.yfus) #how many ribs do we make based on max. allowable battery length
        return [(i+1)/(nribs+1)*y2+(1-(i+1)/(nribs+1))*y1 for i in range(nribs)], nribs
   

    def plot(self, plotter:pv.Plotter, cres:int=9, bres:int=25):
        #1) special points -lg and motors
        ncoords_lgmot_fushn = np.array([
            [self.xle_at_y(self.yfus), self.yfus, self.zle_at_y(self.yfus)],
            [self.xlg, self.ylg, self.zlg],
            [self.xm1, self.ym1, self.zm1],
            [self.xm2, self.ym2, self.zm2],
            [self.xm3, self.ym3, self.zm3],
            [self.xm4, self.ym4, self.zm4],
            [self.xle_at_y(self.yhn), self.yhn, self.zle_at_y(self.yhn)]
        ])

        plotter.add_points(
            ncoords_lgmot_fushn,
            color="red",
            point_size=8,
            render_points_as_spheres=True
        )

        #2) Skins
        xpercs = np.linspace(self.xperc_fore, self.xperc_rear, cres)
        ys = np.linspace(0., self.ytip, bres)
        xpercmesh, ymesh = np.meshgrid(xpercs, ys)
        ids = np.arange(bres*cres).reshape(bres, cres)

        cells = list()
        for i in range(bres-1):
            for j in range(cres-1):
                cells.append([4, ids[i, j], ids[i, j+1], ids[i+1, j+1], ids[i+1, j]])
        cells = np.array(cells).flatten()

        x = np.zeros(bres*cres)
        y = np.zeros(bres*cres)
        ztop = np.zeros(bres*cres)
        zbot = np.zeros(bres*cres)
        for i in range(bres):
            for j in range(cres):
                index = i*cres+j
                x[index] = self.x_from_xperc_reduced(xpercmesh[i, j], ymesh[i, j]) 
                y[index] = ymesh[i, j]
                ztop[index] = self.upper_skin_z(xpercmesh[i, j], ymesh[i, j])
                zbot[index] = self.lower_skin_z(xpercmesh[i, j], ymesh[i, j])
        ncoords_top = np.vstack((x, y, ztop)).T
        ncoords_bot = np.vstack((x, y, zbot)).T

        cell_types = np.full((bres-1)*(cres-1), pv.CellType.QUAD)
        mesh_top = pv.UnstructuredGrid(cells, cell_types, ncoords_top)
        mesh_bot = pv.UnstructuredGrid(cells, cell_types, ncoords_bot)

        plotter.add_mesh(
            mesh_top,
            color="lightblue"
        )

        plotter.add_mesh(
            mesh_bot,
            color="lightblue"
        )

        #3) scaffold
        scaffold_ncoords = self.scaffold.reshape(self.scaffold.shape[0]*self.scaffold.shape[1], self.scaffold.shape[2])

        plotter.add_points(
            scaffold_ncoords,
            color="gray",
            point_size=8,
            render_points_as_spheres=True
        )