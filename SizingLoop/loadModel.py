import numpy as np
import numpy.typing as nt
import funtools as ft

'''Model of the aerodynamic load acting on the wing as eliptically distributed spanwise and NACA2412 cp-distributed chordwise'''
#derivative multiplier means to be multiplied with the toatal lift!

@ft.cache
def per_b_half(yPerB2:nt.NDArray[np.float64]) -> nt.NDArray[np.float64]:
    '''derivative multiplier of Lift force with respect to fraction of halfspan,
    computed from the properties of elliptical distribution'''
    return 8/np.pi*np.sqrt(1-yPerB2**2)

@ft.cache
def per_x_c(xPerC:nt.NDArray[np.float64]) -> nt.NDArray[np.float64]:
    '''derivative multiplier of Lift force with respect to fraction of chord,
    computed from the properties of cp distribution'''
    raise NotImplementedError

def per_xcb2(xPerC:nt.NDArray[np.float64], yPerB2:nt.NDArray[np.float64]) -> nt.NDArray[np.float64]:
    '''derivative multiplier of Lift force with respect to x/c and y/(b/2)'''
    return per_x_c(xPerC)*per_b_half(yPerB2)

#TODO: add resolution parameters
def integrate(xPerC_min:np.float64, xPerC_max:np.float64, yPerB2_min:np.float64, yPerB2_max:np.float64) -> np.float64:
    '''fraction of total Lift force inside an area on the x/c and b/2 grid'''
    raise NotImplementedError
