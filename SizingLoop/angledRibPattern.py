import pointsFromCAD as pfc
import geometricClasses as gcl
import typing as ty

class JointPoints(): 
    def __init__(self, fi:gcl.Point3D, ri:gcl.Point3D, fo:gcl.Point3D, ro:gcl.Point3D): #front-inner/rear-outer
        #TODO: implement a class made of 3d mid points that will represent a very idealised, outer dims only, version of the diagonal rib
        self.fi = fi
        self.ri = ri
        self.fo = fo
        self.ro = ro

def ray_rib_pattern(jointWidth:float, startTop=True, endTop=True) -> ty.Tuple[ty.List[JointPoints], gcl.Direction2D, ty.List[gcl.Line2D]]:
    '''Generates the diagonal ribs. Return points grouped in joint points plus the dihedral vector, upper and lower spanwise directions'''
    #1) create a 2d trapezoid of the wingbox.
    ft = gcl.Point2D(pfc.frt.y, pfc.frt.z) #fuselage top, etc.
    fb = gcl.Point2D(pfc.ffb.y, pfc.ffb.z)
    tt = gcl.Point2D(pfc.tft.y, pfc.tft.z)
    tb = gcl.Point2D(pfc.tfb.y, pfc.tfb.z)

    #2) compute the top, bottom, dihedral vectors and +- 60 deg from dihedral
    topLine = gcl.Line2D.from_pts(ft, tt)
    botLine = gcl.Line2D.from_pts(fb, tb)
    midf = gcl.Point2D.midpoint([ft, fb])
    midt = gcl.Point2D.midpoint([tt, tb])
    dihedralDir = gcl.Direction2D.from_pts(midf, midt)
    ribUp = gcl.Direction2D.rotate(dihedralDir, 60)
    ribDown = gcl.Direction2D.rotate(dihedralDir, -60)

    #3) create the starting condition based on input
    nearPts = list() #here be the 2d projection of each joint closer to root
    farPts = list() #here further from root
    if startTop:
        #creating the first joint
        nearPt = ft
        farPt = topLine.dir.step(nearPt, jointWidth)
        top = True #bool storing if you are currently on top or not
    else:
        nearPt = fb
        farPt = botLine.dir.step(nearPt, jointWidth)
        top = False

    #4) a loop generating next joints until the wing is over
    while farPt.x<=tt.x:
        nearPts.append(nearPt)
        farPts.append(farPt)
        if top: #if you're on the top, tou want to go down
            ribline = gcl.Line2D(farPt, ribDown)
            nearPt = ribline.intersect(botLine)
            farPt = botLine.dir.step(nearPt, jointWidth)
            top=False
        else:
            ribline = gcl.Line2D(farPt, ribUp)
            nearPt = ribline.intersect(topLine)
            farPt = topLine.dir.step(nearPt, jointWidth)
            top=True

    #5) adjusting the joint list for the ending boundary condition
    #5.1) removing the required amount of joints
    if top == endTop: #then the last generated rib was a bottom rib, and we have to end at Top, or top and we have to end at Bot
        nearPts = nearPts[:-2]
        farPts = farPts[:-2]
    else:
        nearPts = nearPts[:-1]
        farPts = farPts[:-1]
    #5.2) creating a new joint that respects the bc in place of the removed ones
    if endTop: #we replace the removed joints
        farPts.append(tt)
        nearPts.append(topLine.dir.step(tt, -jointWidth))
    else:
        farPts.append(tb)
        nearPts.append(botLine.dir.step(tb, -jointWidth))
    
    #6) creating a list of 3D joints out of the 2D representations
    joints = list()
    #6.1) a "top view" of the wing
    ff = gcl.Point2D(pfc.ffb.x, pfc.ffb.y) #fuselage front, etc.
    fr = gcl.Point2D(pfc.frb.x, pfc.frb.y)
    tf = gcl.Point2D(pfc.tfb.x, pfc.tfb.y)
    tr = gcl.Point2D(pfc.trb.x, pfc.trb.y)
    frontLine = gcl.Line2D.from_pts(ff, tf)
    rearLine = gcl.Line2D.from_pts(fr, tr)
    #6.2) a tool for adding the x dimension to joint points
    def x4joint(pt:gcl.Point2D): #pt.x - spanwise, pt.y - height
        chordLine = gcl.Line2D(gcl.Point2D(0, pt.x), gcl.Direction2D(1, 0))
        fore2D = chordLine.intersect(frontLine)
        rear2D = chordLine.intersect(rearLine)
        fore3D = gcl.Point3D(fore2D.x, fore2D.y, pt.y)
        rear3D = gcl.Point3D(rear2D.x, rear2D.y, pt.y)
        return fore3D, rear3D
    #6.3) appending joints using the lists of 2D points
    for i, nearP in enumerate(nearPts):
        farP = farPts[i]
        foreNear, rearNear = x4joint(nearP)
        foreFar, rearFar = x4joint(farP)
        joints.append(JointPoints(foreNear, rearNear, foreFar, rearFar)) 

    return joints, dihedralDir, [topLine, botLine]