import aerosandbox.numpy as np

class Joint():
    def __init__(self, d_inch:float, d_ratio_insert:float, Nmax:float, Vmax:float, SF:float=2., nrows:int=2, bearing_ratio:float=1., sig_b_sheet:float=441e6, rho_bolt:float=8000., rho_insert:float=2780., rho_sheet:float=2780., spacing_ratio:float=3., edge_ratio:float=2.):
        #input data in ich (bolt d), mm (sheet t) and lbf (allowable forces), as taken from the MIL-B-6812E datasheet
        #spacing as load direction from Chapter 4 DOI: https://doi.org/10.1016/B978-0-323-91682-0.00018-2
        #defaults assuming steel bolts, rest from alu 2024-T4. alu from the ususal source,
        #steel density from https://www.greatmetal.com/astm-a193-grade-b7-bolts-fasteners.html
        inch_to_m = 0.0254
        lbf_to_N = 4.44822

        self.d_bolt = d_inch*inch_to_m
        self.d_insert = d_ratio_insert*self.d_bolt
        self.spacing = spacing_ratio*self.d_bolt
        self.edge = edge_ratio*self.d_bolt
        self.Nmax = Nmax*lbf_to_N/SF
        self.Vmax = Vmax*lbf_to_N/SF

        #sheet thickness derivative of bearing to max shear strength
        self.Vbea = bearing_ratio*self.Vmax
        self.t_sheet = self.Vbea/(sig_b_sheet/SF)/self.d_bolt
        
        self.rho_bolt = rho_bolt
        self.rho_insert = rho_insert
        self.rho_sheet = rho_sheet

        self.nrows = nrows #how many rows of bolts there are in the joint

    
    def _ceil_to_int(self, num:int, unit:int):
        mod = num % unit
        if mod == 0:
            return num
        else:
            return num + unit - mod
    
    
    def get_joint_n(self, N:float, V:float):
        n_bolt = int(np.ceil(np.sqrt(N**2/self.Nmax**2+V**2/self.Vmax**2)))
        n_bear = int(np.ceil(V/self.Vbea))
        n_higher = max(n_bolt, n_bear, self.nrows)
        return self._ceil_to_int(n_higher, self.nrows) #to accout for the fact that we need an even number of joints
    

    def get_joint_dims(self, n:int):
        assert n%self.nrows == 0 #as needed for our joint geometry
        nlines = n//self.nrows
        assert nlines >= 1
        return self.spacing*max(nlines-1, self.nrows-1)+2*self.edge #a single row of bolts will have the perpendicular dim longer
    

    def get_joint_mass(self, n:int, H:float, rj:float):
        bolt_area = np.pi*self.d_bolt**2/4
        m_bolt = bolt_area*(H+4*.6*self.d_bolt)*self.rho_bolt #.6 as the bolt head height to diameter taken from 
        m_insert = np.pi*(self.d_insert**2-self.d_bolt**2)/4*H*self.rho_insert
        l1 = 2*self.edge if n==self.nrows else (self.nrows-1)*self.spacing+2*self.edge #to compensate for rjoint
        m_sheet = (l1*rj-n*bolt_area)*self.t_sheet*self.rho_sheet
        return n*(m_bolt+m_insert)+m_sheet
    


class JointsAvailable():
    #NOTE: MIL-B-6812E assumes double shear, we will have single shear so we have to halve the values
    #all bolt fine-threaded steel
    _JOINT_DATA = [Joint(d_inch, inse_ratio, Nmax, Vmax, nrows=nrows, SF=1.5) for d_inch, inse_ratio, Nmax, Vmax, nrows in zip(
       [1/4, 5/16, 3/8, 7/16, 1/2, 9/16, 5/8, 3/4, 7/8, 1, 1+1/8, 1+1/4]*2, #diameters
       [1.25, 1.25, 1.19, 1.19, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]*2, #insert diameter ratios
       [4080, 6500, 10100, 13600, 18500, 23600, 30100, 44000, 60000, 80700, 101800, 130200]*2, #allowable tesile
       [7360/2, 11500/2, 16560/2, 22500/2, 29400/2, 37400/2, 46000/2, 66300/2, 90100/2, 117800/2, 147500/2, 182100/2]*2, #allowable shear
       [2]*12+[4]*12 #whether 2 bolt rows or 4 bolt holes
    )]

    @classmethod
    def size_joint(cls, N:float, V:float, H:float, debug=False):
        ns = list()
        ljs = list()
        mjs = list()

        for joint_candidate in cls._JOINT_DATA:
            n = joint_candidate.get_joint_n(N, V)
            lj = joint_candidate.get_joint_dims(n)
            mj = joint_candidate.get_joint_mass(n, H, lj)
            
            ns.append(n)
            ljs.append(lj)
            mjs.append(mj)
        
        smallest_joint_idx = np.argmin(ljs)
        lj = ljs[smallest_joint_idx]
        mj = mjs[smallest_joint_idx]

        if not debug:
            return (lj, mj)
        
        n = ns[smallest_joint_idx]
        smallest_joint = cls._JOINT_DATA[smallest_joint_idx]

        return (smallest_joint, n, lj, mj)