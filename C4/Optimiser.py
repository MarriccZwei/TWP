import LoadCase as lc
import Pyfe3DModel as p3m
import Solution as sol
import scipy.optimize as opt
import typing as ty

class Optimiser():
    def __init__(self, desvars:ty.Dict[str,float], loadCases:ty.List[ty.Dict[str, object]], cadstrs:ty.Dict[str, str], 
                 materials:ty.Dict[str, float], resConfig:ty.Dict[str, object], g0:float, MTOM:float):
        '''
        The constructor performs the initialisation flow as well
        
        :param desvars: design variables dictionary - keys of 'W_<eleType>' and '(2t/H)_<eleType>'
        :type desvars: ty.Dict[str, float]
        :param loadCases: a dictionary with init data of the load cases
        :type loadCases: ty.List[ty.Dict[str, object]]
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
        '''
        self.desvars = desvars
        self.loadCases = loadCases
        self.cadstrs = cadstrs
        self.materials = materials
        self.resConfig = resConfig
        self.g0 = g0
        self.MTOM = MTOM


    def constraints(self):
        '''Updates the FEM model and weight, and conducts the simulations'''

    def objective(self):
        '''updates the FEM model and obtains the weight'''

    def constraints_for_optim(self):
        ''''''