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
    '''fraction of total Lift force inside an area on the x/c and b/2 grid, and point of application of equivalent force'''
    sideFactor = npeak if top else 1
    arcmin = np.arcsin(yPerB2_min)
    arcmax = np.arcsin(yPerB2_max)
    intx = lambda xPerC:xPerC**3/3-xPerC**2+xPerC
    inty = lambda arcyb:.5*(arcyb+np.sin(2*arcyb)/2) # June 26 page in the booklet
    inx = intx(xPerC_max)-intx(xPerC_min) #saving the coefficient integrals for later
    iny = inty(arcmax)-inty(arcmin)
    integral = 6*sideFactor/(npeak+1)/np.pi*inx*iny # see the booklet page for why is the 6/pi there

    # assert not np.isclose(inx, 0)
    # assert not np.isclose(iny, 0)
    intQx = lambda xPerC:xPerC**4/4-2*xPerC**3/3+xPerC**2/2 #centroid component in x direction
    intQy = lambda yPerB2:-(1-yPerB2**2)**1.5/3 #see June 27 booklet page for derivation
    Qx = intQx(xPerC_max)-intQx(xPerC_min)
    Qy = intQy(yPerB2_max)-intQy(yPerB2_min)

    return integral, Qx/inx, Qy/iny

def apply_on_wingbox(xmesh:nt.NDArray[np.float64], ymesh:nt.NDArray[np.float64], brange:ty.Tuple[float], crange:ty.Tuple[float], 
                     top=True, debug=False):
    '''applying aerodynamic loads on a wing box skin. The pressure in the spanwise/chordwise range of the wingbox
    is distributed over it, the pressure from remaining parts of the wing is integrated and added as an equivalent 
    point load and moment load at the closest node. x-, ymesh are xs and ys of nodes in a 2d array, shape[0] along
    the chord, shape[1] along the span. Transpose the input if necessary to match this requirement! 
    x is chordwise coord, y is spanwise coord'''
    #1) obtaining node coordinates as chord fractions
    ab = (brange[1]-brange[0])/(ymesh[-1,-1]-ymesh[0,0]) #spanwise we need only one proportionality coefficient for all points
    yPerB2 = ab*(ymesh-ymesh[0,0])+brange[0] #linear interpolation to the normalised coordinate

    xPerC = np.zeros(xmesh.shape) #obtaining the chordwise normalised coords is more involved due to taper
    acs = np.zeros(xmesh.shape[1]) #save the slopes for later - for moment arm calculations
    for j in range(xmesh.shape[1]): #in the chordwise direction, we have a different a at each spanwise coordinate
        assert not np.isclose(xmesh[-1,j], xmesh[0,j])
        acs[j] = (crange[1]-crange[0])/(xmesh[-1,j]-xmesh[0,j]) #repeating the procedure for y, but for each chord locally
        xPerC[:,j] = acs[j]*(xmesh[:,j]-xmesh[0,j])+crange[0]

    #2) obtaining a set of node wise normalised half distances (in 4 directions for each node, use 0 at the ends)
    ncxp = np.zeros(xmesh.shape) #normalised coord halfway towards positive x, excluding the boundary zeros
    ncyp = np.zeros(xmesh.shape) #normalised coord halfway towards positive y, excluding the boundary zeros
    ncxm = np.zeros(xmesh.shape) #normalised coord halfway towards negative x, excluding the boundary zeros
    ncym = np.zeros(xmesh.shape) #normalised coord halfway towards negative y, excluding the boundary zeros

    #the boundary points will have their midcoords calculated wrt the 0 or 1. They will cover the entire remaining part of the wing.
    #integration includes obtaining moment, so the node being not in the center of the integral area should not pose an issue 
    xPerCpad = np.pad(xPerC, [(1,0), (0,0)], mode='constant', constant_values=-crange[0]) #so that the bound averages to 0
    xPerCpad = np.pad(xPerCpad, [(0,1),(0,1)], mode='constant', constant_values=2-crange[1]) #so that the abound averages to 1
    yPerB2pad = np.pad(yPerB2, [(0,0),(1,0)], mode='constant', constant_values=-brange[0]) #same story as with xPerCpad
    yPerB2pad = np.pad(yPerB2pad, [(0,0),(0,1)], mode='constant', constant_values=2-brange[1])

    for i in range(xmesh.shape[0]):
        for j in range(xmesh.shape[1]): #TODO: cache this cuz there's some repetition
            ncxp[i,j] = (xPerCpad[i+2,j]+xPerCpad[i+1,j])/2 #there are 2 more indices in i and the middles should align, so i<->i+1
            ncyp[i,j] = (yPerB2pad[i, j+2]+yPerB2pad[i,j+1])/2 #there are 2 more indices in j and the middles should align, so j<->j+1
            ncxm[i,j] = (xPerCpad[i+1,j]+xPerCpad[i,j])/2
            ncym[i,j] = (yPerB2pad[i,j+1]+yPerB2pad[i,j])/2

    #3) integrating node-wise using half of the normalised distance as bounds
    #From applicationPoint, the moment arm (converted to real coords) is obtained.
    f = np.zeros(xmesh.shape)
    Mx = np.zeros(xmesh.shape)
    My = np.zeros(xmesh.shape)
    for i in range(xmesh.shape[0]):
        for j in range(xmesh.shape[1]):
            f[i,j], applXperCs, applYperB2s = integrate(ncxm[i,j], ncxp[i,j], ncym[i,j], ncyp[i,j], top)
            
            #moment arm calculation - positive means arm's direction is towards positive coordinate
            deltay = (applYperB2s-yPerB2[i,j])/ab #inverse of getting from deltay to delta normalised y
            deltax = (applXperCs-xPerC[i,j])/acs[j] #inverse of getting from deltay to delta normalised x

            #moment calculations
            Mx[i,j] = deltay*f[i,j] #jxk=i
            My[i,j] = -deltax*f[i,j] #ixk=-j

    if debug:
        return f, Mx, My, ncxp, ncxm, ncyp, ncym, yPerB2, xPerC

    return f, Mx, My


if __name__ == "__main__":
    #the result should be 0.5, distributed accordingly
    print(integrate(0, 1, 0, 1, True), integrate(0, 1, 0, 1, False), integrate(0, 1, 0, 1, True)[0]+integrate(0, 1, 0, 1, False)[0])
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
    ax = fig.add_subplot(221, projection='3d')
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
    
    import geometricClasses as gcl
    pts1 = [gcl.Point3D(-5, 0, 1), gcl.Point3D(-3, 5, 1.5), gcl.Point3D(-3, 5, 1.5), gcl.Point3D(-2.5, 10, 2)]
    pts2 = [gcl.Point3D(5, 0, 0), gcl.Point3D(3, 5, .5), gcl.Point3D(3, 5, .5), gcl.Point3D(2.5, 10, 1)]
    sheet = gcl.multi_section_sheet3D(pts1, pts2, 9, [7, 2, 11])
    x, y, _ = gcl.pts2coords3D(np.ravel(sheet))
    x, y = np.array(x).reshape(sheet.shape), np.array(y).reshape(sheet.shape)
    f, Mx, My, ncxp, ncxm, ncyp, ncym, yPerB2, xPerC = apply_on_wingbox(x, y, (0.1,0.8), (0.2,0.7), True, True)
    bx = fig.add_subplot(222, projection="3d")
    bx.plot_surface(x, y, f)
    bx.set_title('f')
    cx = fig.add_subplot(223, projection="3d")
    cx.plot_surface(x, y, Mx)
    cx.set_title('M_x')
    dx = fig.add_subplot(224, projection="3d")
    dx.plot_surface(x, y, My)
    dx.set_title('M_y')

    plt.figure()
    for xp, xm, yp, ym, yb, xc in zip(np.ravel(ncxp), np.ravel(ncxm), np.ravel(ncyp), np.ravel(ncym), np.ravel(yPerB2), np.ravel(xPerC)):
        plt.plot([xm, xm, xp, xp, xm], [ym, yp, yp, ym, ym])
        plt.scatter([xc], [yb])
        
    print(sum(np.ravel(f)), sum(np.ravel(Mx)), sum(np.ravel(My)))

    plt.figure()
    levels = np.linspace(f.min(), f.max(), 50)
    plt.contourf(x, y, f, levels=levels)
    plt.colorbar()
    plt.show()