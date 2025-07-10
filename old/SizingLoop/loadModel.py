import numpy as np
import numpy.typing as nt
import funtools as ft

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
    intx = lambda xPerC:xPerC**3/3-xPerC**2+1
    inty = lambda arcyb:.5*(arcyb+np.sin(2*arcyb)/2) # may 18 page in the booklet
    return 3*sideFactor/(npeak+1)/np.pi*((xPerC_max))*(intx(xPerC_max)-intx(xPerC_min))*(inty(arcmax)-inty(arcmin))

if __name__ == "__main__":
    #the result should be 0.5, distributed accordingly
    print(integrate(0, 1, 0, 1, True), integrate(0, 1, 0, 1, False), integrate(0, 1, 0, 1, True)+integrate(0, 1, 0, 1, False))
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
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
    z = np.array(per_xcb2(np.ravel(x), np.ravel(y), False))
    z = z.reshape(x.shape)
    ax.plot_surface(x, y, z)
    plt.show()