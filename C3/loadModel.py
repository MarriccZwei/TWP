import numpy as np
import numpy.typing as nt
import funtools as ft
import typing as ty

'''Model of the aerodynamic load acting on the wing as eliptically distributed spanwise and NACA2412 cp-distributed chordwise'''
#derivative multiplier means to be multiplied with the toatal lift!
npeak = 4 # a parameter required to model the cp distribution

#@ft.cache
def per_b_half(yPerB2:nt.NDArray[np.float64]) -> nt.NDArray[np.float64]:
    '''derivative multiplier of Lift force with respect to fraction of halfspan,
    computed from the properties of elliptical distribution'''
    return 2/np.pi*np.sqrt(1-yPerB2**2)

#@ft.cache
def per_x_c(xPerC:nt.NDArray[np.float64], top:bool) -> nt.NDArray[np.float64]:
    '''derivative multiplier of Lift force with respect to fraction of chord,
    computed from the properties of cp distribution'''
    sideFactor = npeak if top else 1
    return 1.5*sideFactor/(npeak+1)*(xPerC-1)**2

def per_xcb2(xPerC:nt.NDArray[np.float64], yPerB2:nt.NDArray[np.float64], top:bool) -> nt.NDArray[np.float64]:
    '''derivative multiplier of Lift force with respect to x/c and y/(b/2)'''
    return per_x_c(xPerC, top)*per_b_half(yPerB2)

#TODO: add resolution parameters
def integrate(xPerC_min:np.float64, xPerC_max:np.float64, yPerB2_min:np.float64, yPerB2_max:np.float64, top=True) -> np.float64:
    '''fraction of total Lift force inside an area on the x/c and b/2 grid'''
    sideFactor = npeak if top else 1
    arcmin = np.arcsin(yPerB2_min)
    arcmax = np.arcsin(yPerB2_max)
    intx = lambda xPerC:xPerC**3/3-xPerC**2+xPerC
    inty = lambda arcyb:.5*(arcyb+np.sin(2*arcyb)/2) # may 18 page in the booklet
    return 6*sideFactor/(npeak+1)/np.pi*((xPerC_max))*(intx(xPerC_max)-intx(xPerC_min))*(inty(arcmax)-inty(arcmin))

def apply_on_wingbox(xmesh:nt.NDArray[np.float64], ymesh:nt.NDArray[np.float64], brange:ty.Tuple[float], crange:ty.Tuple[float], top=True):
    '''applying aerodynamic loads on a wing box skin. The pressure in the spanwise/chordwise range of the wingbox
    is distributed over it, the pressure from remaining parts of the wing is integrated and added as an equivalent 
    point load and moment load at the closest node. x-, ymesh are xs and ys of nodes in a 2d array, shape[0] along
    the chord, shape[1] along the span. Transpose the input if necessary to match this requirement! '''
    #1) obtaining node coordinates as chord fractions
    #2) obtaining a set of node wise normalised half distances (in 4 directions for each node, use 0 at the ends)
    #2) integrating node-wise using half of the normalised distance as bounds
    #3) going along each of the boundaries, using the array from 2) to and brange or crange to integrate the remaining pressure
    #the point of application is takes as middle of the normalised area. From that moment arm (converted to real coords) is obtained.

if __name__ == "__main__":
    #the result should be 0.5, distributed accordingly
    print(integrate(0, 1, 0, 1, True), integrate(0, 1, 0, 1, False), integrate(0, 1, 0, 1, True)+integrate(0, 1, 0, 1, False))
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    # fig1 = plt.figure()
    # plt.subplot(121)
    # x = np.linspace(0, 1, 100)
    # zt = per_x_c(x, True)
    # zb = per_x_c(x, False)
    # plt.plot(x, zt, label="t")
    # plt.plot(x, zb, label="b")
    # plt.legend()
    # plt.subplot(122)
    # y = np.linspace(0, 1, 100)
    # z = per_b_half(y)
    # plt.plot(y, z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    # z = list()
    # for x_ in x:
    #     row = []
    #     for y_ in y:
    #         row.append(per_xcb2(x, y, True))
    #     z.append(row)
    # z = np.array(z)
    # print(z)
    x, y = np.meshgrid(x, y)
    zb = -np.array(per_xcb2(np.ravel(x), np.ravel(y), False))
    zb = zb.reshape(x.shape)
    ax.plot_surface(x, y, zb)
    zt = np.array(per_xcb2(np.ravel(x), np.ravel(y), True))
    zt = zt.reshape(x.shape)
    ax.plot_surface(x, y, zt)
    plt.show()