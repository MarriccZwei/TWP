import aerosandbox as asb
import aerosandbox.numpy as np
import scipy.interpolate as sip

class ElysianWing():
    def __init__(self, GEOM_SOURCE:dict[str, float | asb.Airfoil], H_per_c_sq:float):
        #0) Overall wing boundaries
        self.yfus = GEOM_SOURCE["yfus"]
        self.yhn = GEOM_SOURCE["yhn"]
        self.ytip = GEOM_SOURCE["ytip"]
        delta_z_hn_fus = GEOM_SOURCE["deltazhn"]
        self._dz_dy = delta_z_hn_fus/(self.yhn-self.yfus)
        delta_x_hn_fus = GEOM_SOURCE["deltaxhn"]
        self._dx_dy = delta_x_hn_fus/(self.yhn-self.yfus)
        self.xtip = self.xle_at_y(self.ytip)
        self.ztip = self.zle_at_y(self.ztip)

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

        #4) The foils for structural centerline
        self.xleroot_reduced = H_per_c_sq/2*self.cr
        self.xletip_reduced = H_per_c_sq/2*self.ct+self.xtip
        self._rootXSec_reduced = asb.WingXSec([self.xleroot_reduced, 0., 0.], self.cr_reduced, 0., GEOM_SOURCE["rootfoil"])
        self._tipXSec_reduced = asb.WingXSec([self.xleroot_reduced, self.ytip, self.ztip], self.ct_reduced, 0., GEOM_SOURCE['tipfoil'])
    

    def c_at_y(self, y:float):
        assert 0.<=y<=self.ytip
        yfrac = y/self.ytip
        return self.cr*(1-yfrac)+self.ct*yfrac
    

    def xle_at_y(self, y:float):
        assert 0.<=y<=self.ytip
        return self._dx_dy*y
    

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


    def _skin_z(self, xperc:float, y:float, upperSide:int): #xperc wrt the reduced chord
        assert 0.<=xperc<=1.
        xsec = self._reduced_xsec_at_y(y)
        foilcoords = xsec.airfoil.coordinates
        foilcoords = foilcoords[:, foilcoords[:, 1]*upperSide>=0.]
        interp = sip.CubicSpline(foilcoords[:, 0], foilcoords[:, 1])
        return xsec.chord*interp(xperc)
        

    def upper_skin_z(self, xperc:float, y:float):
        return self._skin_z(xperc, y, 1)
    
    
    def lower_skin_z(self, xperc:float, y:float):
        return self._skin_z(xperc, y, -1)
    
