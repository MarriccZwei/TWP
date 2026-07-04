from .Geometry.geometryInit import geometry_init
from .LoadCase import LoadCase
from .Solution.processLoadCase import process_load_case, process_aeroelastic_load_case
from .Solution.eleProps import load_ele_props

import typing as ty
import aerosandbox as asb
import aerosandbox.numpy as np
import numpy.typing as nt
import scipy.optimize as opt
import gc

class Optimiser():
    def __init__(self, desvarsInitial:ty.Dict[str,float], loadCasesInfo:ty.List[ty.Dict[str, object]], GEOM_SOURCE:dict[str, float], HYPERPARAMS:dict[str, float], 
                 MASSES:dict[str, float], N:int, materials:ty.Dict[str, float], resConfig:ty.Dict[str, object], g0:float, MTOM:float, nairfs:int,
                 LINBUCKLSF:float, bounds:ty.Tuple[ty.Dict[str, float]], meshMergeDigits:int=8, logEveryNIters:int=None, loadCasesJoint:ty.List[ty.Dict[str, object]]=None):
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
        self.lb = bounds[0]
        self.ub = bounds[1]

        #0) handle logging or lack of thereof
        self.logEveryNIters = logEveryNIters
        if not (logEveryNIters is None):
            self.iteration_number = 0

        if loadCasesJoint is None:
            loadCasesJoint = loadCasesInfo

        #1) geometry initialization
        self.model, self.mesher, self.excl, self.wing, ism = geometry_init(GEOM_SOURCE, HYPERPARAMS, MASSES, N, loadCasesJoint, g0, MTOM, meshMergeDigits, resConfig['sks'])
        airfs, les, tes = self.wing.aero_foils(nairfs)
        self.ep:ty.Dict[str, ty.List[object]] = dict() #element property dict, updated during self._update_model

        #2) design variables initialisation and first model updatate
        self.desvars = {
            '(2t/H)_Sq':0.,
            '(2t/H)_Pq':0.,
            '(2t/H)_Aq':0.,
            'W_bb':0.,
            'W_mb':0.,
            'W_lb':0.,
            'ds':0.,
            'de':0.,
            '(2t/H)_sq':0.,
            '(2t/H)_pq':0.,
            '(2t/H)_aq':0.,
        }
        self._update_model(self.desvarvec(desvarsInitial))
        
        #3) load cases initialisation
        self.lcs:ty.List[LoadCase] = list()
        for lcinfo in loadCasesInfo:
            lc = LoadCase(lcinfo["n"], MTOM, self.model.N, g0, lcinfo["Ttot"], lcinfo["op"], 
                                     les, tes, airfs, resConfig["bres"], resConfig["cres"], resConfig["nneighs"], 
                                     lcinfo["aeroelastic"], lcinfo["nlg"], lcinfo['bank'])
            if lcinfo["aeroelastic"]:
                lc.aerodynamic_matrix(*self.mesher.get_submesh_list(['sq', 'Sq']))
            else:
                lc.apply_aero(*self.mesher.get_submesh_list(['sq', 'Sq']))
                lc.apply_landing(self.mesher.get_submesh('li')[0])
                lc.apply_thrust(self.mesher.get_submesh('mi')[0])
            self.lcs.append(lc)

        #4) saving the masses of all constant inertia st. one can get structural mass in the objective
        self._mn_sum = -ism.tot_joint_mass #we have to include joints as structural mass so we cannot exclude them
        for iv in self.model.inertia_vals:
            self._mn_sum += iv


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
        failure_margins = np.zeros(len(self.lcs) * 2)
        for i, lc in enumerate(self.lcs):
            #1) handling savePath for multiple load case results
            if savePath is None:
                savePathLC = None
            else:
                savePathLC = f"{savePath}LC{i}\\"

            #2.0) separate processing for the 'aeroelastic load cases'
            if lc.aeroelastic:
                # flutterCount = process_aeroelastic_load_case(self.model, lc, plot, savePathLC, self.resConfig["kfl"])
                # if flutterCount>failure_margins[3]:
                #     failure_margins[3] = flutterCount
                raise NotImplementedError("Aeroelastic Load Case deprecated!!!")
                    
            #2) load case (post) processing
            else:
                if self.resConfig["klb"] > .5:
                    raise NotImplementedError("Buckling deprecated!!!")
                lcmargins = process_load_case(self.model, lc, self.materials, self.desvars, self.ep["beamtypes"], self.ep["quadtypes"],
                                            self.excl, plot, savePathLC, self.resConfig["klb"])
                
                #3) assesing whether the load case is constraining and updating failure_margins if so
                # if failure_margins[0]<lcmargins[0]:#quad stresses, the more, the worse
                #     failure_margins[0]=lcmargins[0]
                # if failure_margins[1]<lcmargins[1]:#beam stresses, the more, the worse
                #     failure_margins[1]=lcmargins[1]
                # if failure_margins[2]>lcmargins[2]:#linear buckling load multiplier, the less, the worse
                #     failure_margins[2]=lcmargins[2]
                failure_margins[2*i] = lcmargins[0]
                failure_margins[2*i + 1] = lcmargins[1]

        #4) logging if enabled
        if not (self.logEveryNIters is None):
            if self.iteration_number%self.logEveryNIters==0:
                print(f"Step {self.iteration_number} failure margins: {failure_margins}")
        
        return failure_margins
    

    def forward(self, desvarvec:nt.NDArray[np.float64], plot=False, savePath=None)->nt.NDArray[np.float64]:
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

            #2.0) separate processing for the 'aeroelastic load cases'
            if lc.aeroelastic:
                flutterCount = process_aeroelastic_load_case(self.model, lc, plot, savePathLC, self.resConfig["kfl"])
                if flutterCount>failure_margins[3]:
                    failure_margins[3] = flutterCount
                    
            #2) load case (post) processing
            else:
                lcmargins = process_load_case(self.model, lc, self.materials, self.desvars, self.ep["beamtypes"], self.ep["quadtypes"],
                                            self.excl, plot, savePathLC, self.resConfig["klb"])
                
                #3) assesing whether the load case is constraining and updating failure_margins if so
                if failure_margins[0]<lcmargins[0]:#quad stresses, the more, the worse
                    failure_margins[0]=lcmargins[0]
                if failure_margins[1]<lcmargins[1]:#beam stresses, the more, the worse
                    failure_margins[1]=lcmargins[1]
                if failure_margins[2]>lcmargins[2]:#linear buckling load multiplier, the less, the worse
                    failure_margins[2]=lcmargins[2]

            gc.collect()

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
        self.model.KC0_M_update()
        #W=M@g, with this vector the magnitude of the weight vector will be that of total system mass
        unit_gvect = np.array([0.,0.,1.,0.,0.,0.]*self.model.ncoords.shape[0])
        totmass = np.sum(self.model.M@unit_gvect)
        obj = totmass-self._mn_sum #isolating structural mass from other inertia

        #logging if enabled
        if not (self.logEveryNIters is None):
            if self.iteration_number%self.logEveryNIters==0:
                print(f"Step {self.iteration_number} objective: {obj}")
        
        return obj
    

    def constraint(self)->ty.List[ty.Union[opt.NonlinearConstraint, opt.LinearConstraint]]:
        '''
        the constraints for the optimiser to be evaluated taking the vector of design variables, such as the one from
        self.desvarvec, as input.
        
        :return: the constraint list to pass to the scipy optimizer
        :rtype: List[Union[opt.NonlinearConstraint, opt.LinearConstraint]]
        '''
        epsilon = 1e-5
        comparisonFlutter = .5
        ndesvars = len(self.desvars)
        return [opt.NonlinearConstraint(self.simulate_constraints, np.zeros(len(self.lcs) * 2),
                                       np.ones(len(self.lcs) * 2)),
                opt.LinearConstraint(np.eye(ndesvars), np.ones(ndesvars)*epsilon, #for some margin from divergence
                                     np.ones(ndesvars))] #making use of the fact that the desvarvec is normalised


    def _update_model(self, desvarvec:nt.NDArray[np.float64]):
        #NOTE: update happens only if there is a change to the design variables!!!
        if not np.allclose(desvarvec, self.desvarvec()):
            #1) design variables update
            self.desvars = self.desvars_from_vec(desvarvec)

            #2) model properties update
            ep = load_ele_props(self.desvars, self.materials, self.mesher.eleTypes, self.mesher.eleArgs)
            beamprops = ep["beamprops"]
            beamorients = ep["beamorients"]
            shellprops = ep["shellprops"]
            matdirs = ep["matdirs"]
            inertia_vals = ep["inertia_vals"]
            self.model.load_props(beamprops, beamorients, shellprops, matdirs, inertia_vals)
            self.ep = ep

            #3) updating logging step if logging enabled
            if not (self.logEveryNIters is None):
                self.iteration_number+=1
                print(f"\nStep {self.iteration_number} desvars: {self.desvars}")

    
    ORDER = [
    '(2t/H)_Sq', '(2t/H)_Pq', '(2t/H)_Aq',
    'W_bb', 'W_mb', 'W_lb',
    'ds', 'de',
    '(2t/H)_sq', '(2t/H)_pq', '(2t/H)_aq'
    ]

    def desvarvec(self, vardict:ty.Dict[str, float]=None):
        '''
        Returns the current design variables as a vector to be plugged into scipy optimiser
        '''
        if vardict is None:
            normalise = lambda key:(self.desvars[key]-self.lb[key])/(self.ub[key]-self.lb[key])
        else:
            normalise = lambda key:(vardict[key]-self.lb[key])/(self.ub[key]-self.lb[key])
        return np.array([normalise(key) for key in self.ORDER])
    

    def desvars_from_vec(self, desvarvec:nt.NDArray[np.float64]):
        '''
        Converts the design variables returned by the optimiser to the dictionary format used elsewhere in the code
        
        :param desvarvec: the design variables returned by the optimiser
        :type desvarvec: nt.NDArray[np.float64]
        '''
        return {
            k: self.lb[k] + desvarvec[i] * (self.ub[k] - self.lb[k])
            for i, k in enumerate(self.ORDER)
        }