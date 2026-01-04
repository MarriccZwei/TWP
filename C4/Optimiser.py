from .Geometry.geometryInit import geometry_init
from .LoadCase import LoadCase
from .Solution.processLoadCase import process_load_case
from .Solution.eleProps import load_ele_props

import typing as ty
import aerosandbox as asb
import aerosandbox.numpy as np
import numpy.typing as nt
import scipy.optimize as opt

class Optimiser():
    def __init__(self, desvarsInitial:ty.Dict[str,float], loadCasesInfo:ty.List[ty.Dict[str, object]], cadstrs:ty.Dict[str, str], 
                 materials:ty.Dict[str, float], resConfig:ty.Dict[str, object], g0:float, MTOM:float, airfs:ty.List[asb.Airfoil],
                 LINBUCKLSF:float, WMAX:float, meshMergeDigits:int=3, logEveryNIters:int=None):
        '''
        The constructor performs the initialisation flow as well
        
        :param desvarsInitial: design variables dictionary - keys of 'W_<eleType>' and '(2t/H)_<eleType>' with initial values
        :type desvarsInitial: ty.Dict[str, float]
        :param loadCasesInfo: a dictionary with init data of the load cases
        :type loadCasesInfo: ty.List[ty.Dict[str, object]]
        :param cadstrs: a dictionary of CAD export strings required to initialise the model
        :type cadstrs: ty.Dict[str, str]
        :param materials: a dictionary of material properties in format 'E_ALU':72e9 #[Pa]
        :type materials: ty.Dict[str, float]
        :param resConfig: config settings for aerosandbox and eigenvalue analyses resolutions, key value format
        :type resConfig: ty.Dict[str, object]
        :param g0: gravitational acceleration constant
        :type g0: float
        :param MTOM: aircraft MTOM
        :type MTOM: float
        :param LINBUCKLSF: safety factor for linear buckling
        :type LINBUCKLSF: float
        :param meshMergeDigits: the number of digits Mesher will check to determine whether a certain pair of nodes is coincident
        :type meshMergeDigits: int
        :param logEveryNIters: once in how many iterations one has to log, None mans no logging
        :type logEveryNIters: int
        '''
        self.materials = materials
        self.resConfig = resConfig
        self.LINBUCKLSF = LINBUCKLSF
        self.WMAX = WMAX

        #0) handle logging or lack of thereof
        self.logEveryNIters = logEveryNIters
        if not (logEveryNIters is None):
            self.iteration_number = 0

        #1) geometry initialization
        self.model, self.mesher = geometry_init(cadstrs["mesh"], meshMergeDigits)
        les_flat = np.fromstring(cadstrs["les"], sep=",")
        tes_flat = np.fromstring(cadstrs["tes"], sep=",")
        les = les_flat.reshape((len(les_flat)//3, 3))
        tes = tes_flat.reshape((len(les_flat)//3, 3)) #should have same length
        self.ep:ty.Dict[str, ty.List[object]] #element property dict, updated during self._update_model

        #2) load cases initialisation
        self.lcs:ty.List[LoadCase] = list()
        for lcinfo in loadCasesInfo:
            lc = LoadCase(lcinfo["n"], lcinfo["nlg"], MTOM, self.model.N, g0, lcinfo["Ttot"], lcinfo["op"], 
                                     les, tes, airfs, resConfig["bres"], resConfig["cres"], resConfig["nneighs"],
                                     lcinfo["aeroelastic"])
            if lcinfo["aeroelastic"]:
                lc.aerodynamic_matrix(*self.mesher.get_submesh('sq'))
            else:
                lc.apply_aero(*self.mesher.get_submesh('sq'))
            lc.apply_landing(self.mesher.get_submesh('li')[0])
            lc.apply_thrust(self.mesher.get_submesh('mi')[0])
            self.lcs.append(lc)

        #3) design variables initialisation and first model updatate
        self.desvars = {
            '(2t/H)_sq':0.,
            '(2t/H)_pq':0.,
            '(2t/H)_aq':0.,
            'W_bb':0.,
            'W_mb':0.,
            'W_lb':0.
        }
        self._update_model(np.array([
            desvarsInitial['(2t/H)_sq'],
            desvarsInitial['(2t/H)_pq'],
            desvarsInitial['(2t/H)_aq'],
            desvarsInitial['W_bb'],
            desvarsInitial['W_mb'],
            desvarsInitial['W_lb'],
        ]))


    def simulate_constraints(self, desvarvec:nt.NDArray[np.float64], plot=False, savePath=None)->nt.NDArray[np.float64]:
        '''
        Updates the FEM model if necessary and calculates the failure margins based on the provided load cases
        
        :param desvarvec: vector of design variables, in the format required by the optimiser
        :type desvarvec: nt.NDArray[np.float64]
        :param plot: vector of design variables
        :param savePath: the path to which the plots of load case processing results are to be saved
        :return: the array of failure margins: [quad stress, beam stress, load multiplier, complex eigenfrequencies count]
        :rtype: NDArray[float64]
        '''
        self._update_model(desvarvec)
        failure_margins = np.array([0., 0., np.inf, 0.])
        for i, lc in enumerate(self.lcs):
            #1) handling savePath for multiple load case results
            if savePath is None:
                savePathLC = None
            else:
                savePathLC = f"{savePath}LC{i}\\"

            #2) load case (post) processing
            lcmargins = process_load_case(self.model, lc, self.materials, self.desvars, self.ep["beamtypes"], self.ep["quadtypes"],
                                          plot, savePathLC, self.resConfig["kfl"], self.resConfig["klb"])
            
            #3) assesing whether the load case is constraining and updating failure_margins if so
            if failure_margins[0]<lcmargins[0]:#quad stresses, the more, the worse
                failure_margins[0]=lcmargins[0]
            if failure_margins[1]<lcmargins[1]:#beam stresses, the more, the worse
                failure_margins[1]=lcmargins[1]
            if failure_margins[2]>lcmargins[2]:#linear buckling load multiplier, the less, the worse
                failure_margins[2]=lcmargins[2]
            if failure_margins[3]<lcmargins[3]:#complex eigenfrequencies count, the more, the worse
                failure_margins[3]=lcmargins[3]

        #4) logging if enabled
        if not (self.logEveryNIters is None):
            if self.iteration_number%self.logEveryNIters==0:
                print(f"Step {self.iteration_number} failure margins: {failure_margins}")
        
        return failure_margins
            

    def objective(self, desvarvec:nt.NDArray[np.float64])->float:
        '''
        Computes the total mass of the system to be used as an objective to minimise
        
        :param desvarvec: vector of design variables, in the format required by the optimiser
        :type desvarvec: nt.NDArray[np.float64]
        :return: the total mass
        :rtype: float
        '''
        self._update_model(desvarvec)
        #W=M@g, with this vector the magnitude of the weight vector will be that of total system mass
        unit_gvect = np.array([0.,0.,1.,0.,0.,0.]*self.model.ncoords.shape[0])
        totmass = np.sum(self.model.M@unit_gvect)

        #logging if enabled
        if not (self.logEveryNIters is None):
            if self.iteration_number%self.logEveryNIters==0:
                print(f"Step {self.iteration_number} objective: {totmass}")
        
        return totmass
    

    def constraint(self)->ty.List[ty.Union[opt.NonlinearConstraint, opt.LinearConstraint]]:
        '''
        the constraints for the optimiser to be evaluated taking the vector of design variables, such as the one from
        self.desvarvec, as input.
        
        :return: the constraint list to pass to the scipy optimizer
        :rtype: List[Union[opt.NonlinearConstraint, opt.LinearConstraint]]
        '''
        epsilon = 1e-3
        comparisonFlutter = .5
        ndesvars = len(self.desvars)
        return [opt.NonlinearConstraint(self.simulate_constraints, np.array([0.,0., self.LINBUCKLSF, -epsilon]),
                                       np.array([1., 1., np.inf, comparisonFlutter])),
                opt.LinearConstraint(np.eye(ndesvars), np.ones(ndesvars)*epsilon, 
                                     np.array([1., 1., 1., self.WMAX, self.WMAX, self.WMAX]))]


    def _update_model(self, desvarvec:nt.NDArray[np.float64]):
        #NOTE: update happens only if there is a change to the design variables!!!
        if not np.allclose(desvarvec, self.desvarvec()):
            #1) design variables update
            self.desvars = Optimiser.desvars_from_vec(desvarvec)

            #2) model properties update
            ep = load_ele_props(self.desvars, self.materials, self.mesher.eleTypes, self.mesher.eleArgs)
            beamprops = ep["beamprops"]
            beamorients = ep["beamorients"]
            shellprops = ep["shellprops"]
            matdirs = ep["matdirs"]
            inertia_vals = ep["inertia_vals"]
            self.model.KC0_M_update(beamprops, beamorients, shellprops, matdirs, inertia_vals)
            self.ep = ep

            #3) updating logging step if logging enabled
            if not (self.logEveryNIters is None):
                self.iteration_number+=1
                print(f"\nStep {self.iteration_number} desvars: {self.desvars}")

    
    def desvarvec(self):
        '''
        Returns the current design variables as a vector to be plugged into scipy optimiser
        '''
        return np.array([
            self.desvars['(2t/H)_sq'],
            self.desvars['(2t/H)_pq'],
            self.desvars['(2t/H)_aq'],
            self.desvars['W_bb'],
            self.desvars['W_mb'],
            self.desvars['W_lb'],
        ])
    

    @staticmethod
    def desvars_from_vec(desvarvec:nt.NDArray[np.float64]):
        '''
        Converts the design variables returned by the optimiser to the dictionary format used elsewhere in the code
        
        :param desvarvec: the design variables returned by the optimiser
        :type desvarvec: nt.NDArray[np.float64]
        '''
        return {
            '(2t/H)_sq':desvarvec[0],
            '(2t/H)_pq':desvarvec[1],
            '(2t/H)_aq':desvarvec[2],
            'W_bb':desvarvec[3],
            'W_mb':desvarvec[4],
            'W_lb':desvarvec[5]
            }