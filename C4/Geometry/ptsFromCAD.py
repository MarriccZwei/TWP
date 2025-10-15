from .. import geometricClasses as gcl
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
        #estimator of chord length at y
        self.c_at_y = lambda y:self.outteline.for_y(y).x-self.outleline.for_y(y).x
