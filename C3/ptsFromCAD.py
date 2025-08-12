import geometricClasses as gcl
import typing as ty
import numpy as np
import scipy.interpolate as si

def decode(inpt:str)->ty.List[ty.List[gcl.Point3D]]:
    #MILIMETERS->METERS
    return [[gcl.Point3D(*[float(k)/1000 for k in ele.split(';')]) for ele in substr.split('|')] for substr in inpt.split('&')]

def regain_surface(pts1:ty.List[gcl.Point3D], pts2:ty.List[gcl.Point3D], n=8):
    #must be a smooth, function-like surface, one value of z for each x and y pair
    #regains a smooth surface made by cubic interpolation and linear from the airfoil points provided
    #to obtain this we use our knowledge of the geometry -1st we obtain relative choor
    all_pts = []
    for p1, p2 in zip(pts1, pts2):
        all_pts += p1.pts_between(p2, n) #4 so that all coefficients of cubic spline get linear
    x,y,z = gcl.pts2coords3D(all_pts)
    return si.CloughTocher2DInterpolator(list(zip(x, y)), z)
     

def divide_foil(foil:ty.List[gcl.Point3D])->ty.Tuple[ty.List[gcl.Point3D]]:
    #divides the foil into top and bottom segments
    #division happens over the zero segment, the foils are formatted so that they start from the TE of lower part.
    i_origin = np.argmin(np.array([p.x for p in foil])) #origin is the point of lowest x for our coords
    lower = foil[:i_origin+1]
    upper = foil[i_origin:] 
    #making sure that bothe sequences are incresing in x
    lower = lower if lower[0].x<lower[1].x else list(reversed(lower))
    upper = upper if upper[0].x<upper[1].x else list(reversed(upper))
    return upper, lower #important - return order

class UnpackedPoints():
    #how the output of points from CAD is structured
    def __init__(self, inpt:str):
        pts = decode(inpt)
        rt, rb = divide_foil(pts[0])
        tt, tb = divide_foil(pts[1])
        self.surft = regain_surface(rt, tt)
        self.surfb = regain_surface(rb, tb)
        self.motors = pts[2]
        self.hinge = pts[3][0]
        self.lg = pts[3][1]
        self.leline = gcl.Line3D.from_pts(pts[4][0], pts[4][1])
        self.teline = gcl.Line3D.from_pts(pts[5][0], pts[5][1])
        #fuselage cross section f/r - fore rear, t/b - top/bottom
        self.ffb = pts[6][0] #extreme pts for backwards compatibility
        self.frb = pts[6][-2]
        self.frt = pts[6][-1]
        self.fft = pts[6][1]
        self.fcps = pts[6] #cross section points for fuselage section
        #tip cross section - same naming convention
        self.tfb = pts[7][0]
        self.trb = pts[7][-2]
        self.trt = pts[7][-1]
        self.tft = pts[7][1]
        self.tcps = pts[7] #cross section points for tip section
        #x boundaries of the wing
        ttxmax = max([pt.x for pt in tt])
        ttxmin = min([pt.x for pt in tt])
        #the overall chord length same for top/bot, but we have more space taken at the bot
        xc = lambda x:(x-ttxmin)/(ttxmax-ttxmin)
        self.xcft = xc(self.tft.x)
        self.xcrt = xc(self.trt.x)
        self.xcfb = xc(self.tfb.x)
        self.xcrb = xc(self.trb.x)
        #the root and tip les and tes
        self.rle = pts[8][0]
        self.rte = pts[8][1]
        self.tle = pts[8][2]
        self.tte = pts[8][3]
        self.outleline = gcl.Line3D.from_pts(self.rle, self.tle)
        self.outteline = gcl.Line3D.from_pts(self.rte, self.tte)

    

    
if __name__ == "__main__":
    data = "5000;0;-6.5|4750;0;-24|4500;0;-41|4000;0;-75|3500;0;-107|3000;0;-138|2500;0;-167|2000;0;-190|1500;0;-206|1250;0;-211|1000;0;-211.5|750;0;-205|500;0;-187.5|375;0;-173|250;0;-150.5|125;0;-113.5|62.5;0;-82.5|0;0;0|62.5;0;107.5|125;0;149.5|250;0;206.5|375;0;248|500;0;281.5|750;0;330.5|1000;0;363|1250;0;383.5|1500;0;394|2000;0;390|2500;0;362|3000;0;318|3500;0;259|4000;0;187.5|4500;0;104|4750;0;57|5000;0;6.5&4250;18000;2206.872|4125;18000;2198.122|4000;18000;2189.622|3750;18000;2172.622|3500;18000;2156.622|3250;18000;2141.122|3000;18000;2126.622|2750;18000;2115.122|2500;18000;2107.122|2375;18000;2104.622|2250;18000;2104.372|2125;18000;2107.622|2000;18000;2116.372|1937.5;18000;2123.622|1875;18000;2134.872|1812.5;18000;2153.372|1781.25;18000;2168.872|1750;18000;2210.122|1781.25;18000;2263.872|1812.5;18000;2284.872|1875;18000;2313.372|1937.5;18000;2334.122|2000;18000;2350.872|2125;18000;2375.372|2250;18000;2391.622|2375;18000;2401.872|2500;18000;2407.122|2750;18000;2405.122|3000;18000;2391.122|3250;18000;2369.122|3500;18000;2339.622|3750;18000;2303.872|4000;18000;2262.122|4125;18000;2238.622|4250;18000;2213.372&-471.576;3673.46;441.249|126.713;7770.33;945.522|725.002;11867.2;1449.796|1323.292;15964.069;1954.07&3012.266;18950.167;2324.854|4861.588;5721.895;702.56&500;0;0|1999.77;18000;2214.157&3750;0;0|3624.77;18000;2214.157&871.849;1600;0.583|871.849;1600;512.208|1171.071;1600;-6.06|1503.866;1600;570.358|1829.363;1600;6.581|2152.161;1600;565.683|2460.005;1600;32.482|2745.201;1600;526.456|3012.213;1600;63.976|3249.622;1600;475.18|3470.631;1600;92.38|3470.631;1600;447.784&2124.805;18000;2107.631|2124.805;18000;2375.342|2281.374;18000;2104.155|2455.511;18000;2405.769|2625.829;18000;2110.769|2794.735;18000;2403.323|2955.816;18000;2124.322|3105.047;18000;2382.797|3244.763;18000;2140.802|3368.988;18000;2355.966|3484.632;18000;2155.664|3484.632;18000;2341.631&0;0;0|5000;0;0|1749.77;18000;2214.157|4250;18000;2210.122" 
    pts = decode(data)
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.subplot(121, projection='3d')
    ax.plot(*gcl.pts2coords3D(pts[0]), label="root")
    ax.plot(*gcl.pts2coords3D(pts[1]), label="tip")
    ax.scatter(*gcl.pts2coords3D(pts[2]), label="motors")   
    ax.scatter(*gcl.pts2coords3D([pts[3][0]]), label="hinge") 
    ax.scatter(*gcl.pts2coords3D([pts[3][1]]), label="lg") 
    ax.plot(*gcl.pts2coords3D(pts[4]), label="le")
    ax.plot(*gcl.pts2coords3D(pts[5]), label="te")
    ax.plot(*gcl.pts2coords3D(pts[6]), label="fus")
    ax.plot(*gcl.pts2coords3D(pts[7]), label="tt")
    

    bx = plt.subplot(122, projection='3d')
    up = UnpackedPoints(data)
    xs = np.linspace(0, 5, 1000)
    ys = np.linspace(0, 18, 1000)
    X, Y = np.meshgrid(xs, ys)
    bx.plot_surface(X, Y, up.surft(X,Y), label="top_foil")
    bx.plot_surface(X,Y, up.surfb(X,Y), label="bot_foil")
    bx.scatter(*gcl.pts2coords3D(up.motors))
    bx.scatter(*gcl.pts2coords3D([up.hinge, up.lg]))
    bx.plot(*gcl.pts2coords3D(up.fcps))
    bx.plot(*gcl.pts2coords3D(up.tcps))
    lepts = [up.leline.for_y(yval) for yval in ys]
    plt.plot(*gcl.pts2coords3D(lepts), color="black")
    tepts = [up.teline.for_y(yval) for yval in ys]
    plt.plot(*gcl.pts2coords3D(tepts), color="brown")
    plt.plot(*gcl.pts2coords3D([up.rle, up.rte, up.tte, up.tle, up.rle]), color="black")

    plt.legend()
    plt.show()