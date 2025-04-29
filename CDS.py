
trapVertEnc = "2196.48;5;266.241|2196.48;5;-107.114|2196.48;1400;424.795|2196.48;1400;32.391|2196.48;17995;2018.516|2196.48;17995;1841.318"
motor_yzones = "16019.787;16322.714|12263.436;12566.363|8507.085;8810.012|4750.734;5053.661"
lg_yzone = "6409.425;7136.436"
dihedral = "6deg"
LEenc = "631.798;1400;424.795"
TEenc = "3183.4;1400;424.795"
fsang = "2.939deg"
rsang = "179.054deg"
startTop=True
removeLast=True
xlg = 3005.513
cell_d = 21.22
delta = 0.58
cell_h = 70.3

import matplotlib.pyplot as plt

'''COPY-PASTE THIS'''
thicknesses = [1+2**(-.005*i) for i in range(200)]
jointDepth = [50+20**(-0.01*i) for i in range(200)] #[mm]
nteeth = [3 for i in range(25)] + [2 for i in range(175)]
toothDepth = [7 for i in range(25)] + [5 for i in range(175)] #[mm]
adhesive_t = [1 for i in range(50)] + [0.5 for i in range(150)]
#!IMPROVISED!
delta_radius = (cell_d+delta)/2
delta_diameter = cell_d+delta
rect_joint_t = 25
fus_skin_offset = 15
root_rib_offset = 10

import math
class Point2D():
    def __init__(self, y:float, z:float, top:bool):
        self.y = y
        self.z = z
        #btw, the line below is an SRP breach and it will definitely hurt us : (
        self.top = top #true if it's a top point, false if it's a bottom point

    def yin(self, interval):
        return (self.y>=interval[0]) and (self.y<=interval[1])
    
    def zin(self, interval):
        return (self.z>=interval[0]) and (self.z<=interval[1])
    
    def dist(self, pt):
        return math.sqrt((self.y-pt.y)**2+(self.z-pt.z)**2)

class UnitVector2D():
    def __init__(self, delta_y, delta_z):
        if not (delta_y or delta_z): #allowing for a zero vector
            self.y = 0
            self.z = 0
        else:
            magn_delta = math.sqrt(delta_y**2+delta_z**2)
            self.y = delta_y/magn_delta
            self.z = delta_z/magn_delta

    @classmethod #used to obtain directions between adjacent points
    def from_pts(cls, p1:Point2D, p2:Point2D):
        delta_y = p2.y-p1.y
        delta_z = p2.z-p1.z
        return cls(delta_y, delta_z)
    
    @classmethod
    def normal(cls, vec, up:bool):
        #horizontal line protection
        if vec.z ==0:
            return cls(0, 1) if up else cls(0, -1)
        #1st dot product equation for y set to one
        y = 1
        z = -vec.y/vec.z
        #inverting the vector orientation if necessary
        if z<0 != up:
            z *= -1
            y = -1
        return cls(y, z)
    
    @classmethod
    def normal_towards(cls, vec, target):
        if vec.z == 0:
            #horizontal line protection
            y1st = 0
            z1st = 1
            y2nd = 0
            z2nd = -1
        else:
            #1st dot product equation for y set to one
            y1st = 1
            z1st = -vec.y/vec.z
            y2nd = -1
            z2nd = vec.y/vec.z
        
        #normalizing the vectors and taking dot product to find angles
        first = cls(y1st, z1st)
        second = cls(y2nd, z2nd)
        first_cosine = first.dot(target) #for 2 co-oriented normal vects we can get more than 1. (long live float errors) so it's safer not to invert cosines
        second_cosine = second.dot(target)

        #returning the one closer to the target direction
        return first if first_cosine > second_cosine else second

    
    def step(self, pt:Point2D, dist:float): #moving a point a certain distance along the direction
        return Point2D(pt.y+self.y*dist, pt.z+self.z*dist, pt.top)
    
    def dot(self, vec):
        return self.y*vec.y+self.z*vec.z
    
    def rotate(self, angle, deg=False):
        if deg:
            angle = math.radians(angle)
        return UnitVector2D(self.y*math.cos(angle)-self.z*math.sin(angle), self.y*math.sin(angle)+self.z*math.cos(angle))
    
class Joint():
    def __init__(self, p1:Point2D, pt2:Point2D, trigpt:Point2D):
        self.p1 = p1
        self.p2 = pt2
        self.trigpt = trigpt

    #creates a joint based on a point and a set of directions - only if neither r1 or r2 are parallel to c so only for nonzero dihedral
    @classmethod
    def standard(cls, point:Point2D, t1:float, t2:float, c:UnitVector2D, dir1:UnitVector2D, dir2:UnitVector2D, r1:UnitVector2D, r2:UnitVector2D):
            #implements the set of eqns from Apr 26 book
            alph1 = math.acos(abs(dir1.dot(r1)))
            alph2 = math.acos(abs(dir2.dot(r2)))
            p1 = dir1.step(point, t1/2/math.sin(alph1))
            p2 = dir2.step(point, t2/2/math.sin(alph2))
            #finding the intersection point
            delta_dir = (t1/math.cos(alph1)-t2/math.cos(alph2))/(math.tan(alph2)+math.tan(alph1))/2 #distance along dir2
            h = t1/math.cos(alph1)/2-delta_dir*math.tan(alph1) #distance along c
            trigpt = dir2.step(c.step(point, h), delta_dir)
            return cls(p1, p2, trigpt)
    
    @classmethod #as on Apr 27 page
    def edge1(cls, point:Point2D, t1:float, c:UnitVector2D, dir1:UnitVector2D,r1:UnitVector2D):
        alph1 = math.acos(abs(dir1.dot(r1)))
        p1 = dir1.step(point, -t1/2/math.sin(alph1)+t1*math.sin(alph1))
        p2 = dir1.step(point, -t1/2/math.sin(alph1))
        trigpt = c.step(p2, t1*math.cos(alph1))
        return cls(p1, p2, trigpt)

    @classmethod #as on Apr 27 page
    def edge2(cls, point:Point2D, t2:float, c:UnitVector2D, dir2:UnitVector2D,r2:UnitVector2D):
        alph2 = math.acos(abs(dir2.dot(r2)))
        p1 = dir2.step(point, t2/2/math.sin(alph2)-t2*math.sin(alph2))
        p2 = dir2.step(point, t2/2/math.sin(alph2))
        trigpt = c.step(p1, t2*math.cos(alph2))
        return cls(p1, p2, trigpt)

class Line():
    def __init__(self, pt:Point2D, dir:UnitVector2D):
        self.pt = pt
        self.dir = dir

    # def __call__(self, y):
    #     return self.a*y+self.b
    
    def intersect(self, line):
        m = (-line.dir.z*(line.pt.y-self.pt.y)+line.dir.y*(line.pt.z-self.pt.z))/(-self.dir.y*line.dir.z+self.dir.z*line.dir.y)
        return self.dir.step(self.pt, m)
    
    def sidecheck(self, point:Point2D, side:UnitVector2D): #true if on the side of the line pointed by the side vector false otherwise
        point_projection = self.intersect(Line(point, side)) #the target point projected onto the line
        towards_vec = UnitVector2D.from_pts(point_projection, point) #normal vector towards the point

        # proj_stepped = towards_vec.step(point_projection, 25)
        # plt.plot([point_projection.y, proj_stepped.y], [point_projection.z, proj_stepped.z])
        # plt.scatter(point_projection.y, point_projection.z)
        
        return towards_vec.dot(side)>0 #checking if the fectors face towards the same side

    @classmethod
    def from_pts(cls, p1:Point2D, p2:Point2D):
        return cls(p1, UnitVector2D.from_pts(p1, p2))
        
class Wedge():
    @staticmethod
    def adh_offset(main_vertex:Point2D, side_vertex1:Point2D, side_vertex2:Point2D, adhesive_t1, adhesive_t2):
        #adjusting the side vertices for adhesive thickness
        lin_vec = UnitVector2D.from_pts(side_vertex1, side_vertex2) #direction between side vertices
        main_from_1 = UnitVector2D.from_pts(side_vertex1, main_vertex)
        normal_side1 = UnitVector2D.normal_towards(main_from_1, lin_vec)
        #as we want the other normal vector to face also inward and inward, from the other side, is in the other direction
        minuslinvec = UnitVector2D.from_pts(side_vertex2, side_vertex1) 
        main_from_2 = UnitVector2D.from_pts(side_vertex2, main_vertex)
        normal_side2 = UnitVector2D.normal_towards(main_from_2, minuslinvec)
        #actually making the step for adhesive thickness
        cos1 = normal_side1.dot(lin_vec)
        t1 = adhesive_t1/cos1
        side_vertex1 = lin_vec.step(side_vertex1, t1)
        cos2 = normal_side2.dot(minuslinvec)
        t2 = adhesive_t2/cos2
        side_vertex2 = minuslinvec.step(side_vertex2, t2)
        #updating main_vertex for adhesive thickness offset
        main_vertex = Line(side_vertex1, main_from_1).intersect(Line(side_vertex2, main_from_2))
        return main_vertex, side_vertex1, side_vertex2, lin_vec

    @classmethod
    def toothless(cls, main_vertex:Point2D, side_vertex1:Point2D, side_vertex2:Point2D, adhesive_t1, adhesive_t2):
        # #adjusting the side vertices for adhesive thickness
        main_vertex, side_vertex1, side_vertex2, _ = cls.adh_offset(main_vertex, side_vertex1, side_vertex2, adhesive_t1, adhesive_t2)
        return cls([side_vertex1, side_vertex2], main_vertex)

    @classmethod
    def toothed(cls, main_vertex:Point2D, side_vertex1:Point2D, side_vertex2:Point2D, nteeth:float, toothDepth:float, adhesive_t1, adhesive_t2):
        # #adjusting the side vertices for adhesive thickness
        main_vertex, side_vertex1, side_vertex2, lin_vec = cls.adh_offset(main_vertex, side_vertex1, side_vertex2, adhesive_t1, adhesive_t2)

        #if the main vertex is above the midpoint of the tooth curve, the tooth curve will face up, and vice versa
        #this way, the tooth curve will always face towards the main vertex, so never deeper into the battery than thickness depth
        side_dist = math.sqrt((side_vertex1.y-side_vertex2.y)**2+(side_vertex1.z-side_vertex2.z)**2)
        inward_vec = UnitVector2D.normal_towards(lin_vec, UnitVector2D.from_pts(side_vertex1, main_vertex))

        #dividing the curve
        ncurves = 2*nteeth
        curve_len = side_dist/ncurves

        #generating the teeth
        toothpts = [side_vertex1]
        stepup = True
        for i in range(ncurves-1): #for the intermediate points
            pt = lin_vec.step(toothpts[-1], curve_len)
            if stepup: #moving every second point also inwards
                pt = inward_vec.step(pt, toothDepth)
                stepup = False
            else:
                pt = inward_vec.step(pt, -toothDepth) #we have to go back to the outer line, right?
                stepup = True
            toothpts.append(pt) #once we are done with everything
        toothpts.append(side_vertex2) #add it directly not to lose closure from rounding
        return cls(toothpts, main_vertex)

    def __init__(self, toothpts, main_vertex):
        #actual constructor
        self.toothpts = toothpts
        self.main_vertex = main_vertex
        #will be specified later based on triangle lists
        self.xLE=None
        self.xTE=None


#list of wedges that shall get appended by triangles
wedges = list()

class Triangle():
    @staticmethod
    def _tooth_generation(main_vertex:Point2D, side_vertex1:Point2D, side_vertex2:Point2D, nteeth:float, toothDepth:float):
        #if the main vertex is above the midpoint of the tooth curve, the tooth curve will face up, and vice versa
        #this way, the tooth curve will always face towards the main vertex, so never deeper into the battery than thickness depth
        lin_vec = UnitVector2D.from_pts(side_vertex1, side_vertex2) #direction between side vertices
        side_dist = math.sqrt((side_vertex1.y-side_vertex2.y)**2+(side_vertex1.z-side_vertex2.z)**2)
        inward_vec = UnitVector2D.normal_towards(lin_vec, UnitVector2D.from_pts(side_vertex1, main_vertex))

        #dividing the curve
        ncurves = 2*nteeth
        curve_len = side_dist/ncurves

        #generating the teeth
        toothpts = [side_vertex1]
        stepup = True
        for i in range(ncurves-1): #for the intermediate points
            pt = lin_vec.step(toothpts[-1], curve_len)
            if stepup: #moving every second point also inwards
                pt = inward_vec.step(pt, toothDepth)
                stepup = False
            else:
                pt = inward_vec.step(pt, -toothDepth) #we have to go back to the outer line, right?
                stepup = True
            toothpts.append(pt) #once we are done with everything
        toothpts.append(side_vertex2) #add it directly not to lose closure from rounding

        return toothpts

    @staticmethod
    def _three_line_shrink(a:Point2D, b:Point2D, c:Point2D, d:Point2D): #returns a part of the boundary list of the centers polygon generated from the passed 4 points
        #April 28 page
        #shrink direction vectors e.g. abn vector normal to ab in the direction of shrink
        abn = UnitVector2D.normal_towards(UnitVector2D.from_pts(a, b), UnitVector2D.from_pts(a, d))
        bcn = UnitVector2D.normal_towards(UnitVector2D.from_pts(b, c), UnitVector2D.from_pts(b, d))
        cdn = UnitVector2D.normal_towards(UnitVector2D.from_pts(c, d), UnitVector2D.from_pts(c, a))

        #moving the points in their respective shrink directions
        offset = delta+cell_d/2
        a_abn = abn.step(a, offset)
        b_abn = abn.step(b, offset)
        b_bcn = bcn.step(b, offset)
        c_bcn = bcn.step(c, offset)
        c_cdn = cdn.step(c, offset)
        d_cdn = cdn.step(d, offset)

        #computing offset lines
        abl = Line.from_pts(a_abn, b_abn)
        bcl = Line.from_pts(b_bcn, c_bcn)
        cdl = Line.from_pts(c_cdn, d_cdn)

        #computing the intersects
        ab = abl.intersect(bcl)
        bc = abl.intersect(cdl)
        cd = bcl.intersect(cdl)

        #based on the direction of bc relative to bcl versus the bc offset direction, which points are taken is chosen
        towards_bc = UnitVector2D.normal_towards(UnitVector2D.from_pts(ab, cd), UnitVector2D.from_pts(ab, bc))
        direction_cosine = towards_bc.dot(bcn) #due to rounding errors for 2 unit vectors in the same dir this can be > 1. As such acosing it is not possible
        if direction_cosine > 0: #we should never get close to boundary, it should be either 0 or 180 (or almost that cuz floats)
            return [bc]
        else:
            return [ab, cd]


    @staticmethod
    def _populate(v12:Point2D, v13:Point2D, v21:Point2D, v23:Point2D, v31:Point2D, v32:Point2D):
        #calculating the center boundary
        center_boundary = list()
        center_boundary+=Triangle._three_line_shrink(v13, v31, v32, v23)
        center_boundary+=Triangle._three_line_shrink(v32, v23, v21, v12)
        center_boundary+=Triangle._three_line_shrink(v21, v12, v13, v31)
        
        #orienting the boundary - calculating centroid
        npts = len(center_boundary)
        ysum, zsum = 0, 0
        for pt in center_boundary:
            ysum+=pt.y
            zsum+=pt.z
        centroid = Point2D(ysum/npts, zsum/npts, None)

        #calculating inwards normal vectors and boundary lines
        normal_vectors = list()
        lines = list()
        looped_center_boundary = [center_boundary[-1]]+center_boundary #extending the centroid list to calculate normal vects for every edge, starting with a big wall
        for i in range(npts):
            normal_vectors.append(UnitVector2D.normal_towards(UnitVector2D.from_pts(looped_center_boundary[i], looped_center_boundary[i+1]), UnitVector2D.from_pts(looped_center_boundary[i], centroid)))
            lines.append(Line.from_pts(looped_center_boundary[i], looped_center_boundary[i+1]))

        #cell creation -starting at one of the edges -v13, v31
        cellpts = list()
        cellptsrow = list() #here be cells from the current row, which after checking for inside/outside will be added to correctedRow
        correctedRow = list() #after branching into other points will be added to cellpts
        width1strow = looped_center_boundary[0].dist(looped_center_boundary[1])
        ncells1stRow = 1+int(width1strow/delta_diameter)
        along_vec = UnitVector2D.from_pts(looped_center_boundary[0], looped_center_boundary[1])
        integer_margin = (width1strow-(ncells1stRow-1)*delta_diameter)/2 #to center the cell pattern
        #when you are on the top, the directions invert
        if normal_vectors[0].z<0:
            plus30dir = normal_vectors[0].rotate(30, True)
            minus30dir = normal_vectors[0].rotate(-30, True)
        else:
            plus30dir = normal_vectors[0].rotate(-30, True)
            minus30dir = normal_vectors[0].rotate(30, True)
        for i in range(ncells1stRow):
            correctedRow.append(along_vec.step(looped_center_boundary[0], integer_margin+i*delta_diameter)) #here we know that this point is correct

        #cell creation - remaining layers
        while correctedRow != []:
        #for i in range(20): #4 tests
            cellptsrow.append(minus30dir.step(correctedRow[0], delta_diameter)) #generating the one point that can only be created by going in the - dir
            for pt in correctedRow: #the remaining points can be created by going in the plus direction
                cellptsrow.append(plus30dir.step(pt, delta_diameter))            
            #moving the correctedRow to cellpts once their job is done
            cellpts += correctedRow
            correctedRow = list()
            #checking if the generated points are inside
            for pt in cellptsrow:
                inside = True
                for index, edge in enumerate(lines):
                    if not edge.sidecheck(pt, normal_vectors[index]):
                        inside = False
                        #break
                if inside: #if the point is inside, it gets added to the next propagation list
                    correctedRow.append(pt)
            cellptsrow = list() #clearing the row list

        return cellpts



    def __init__(self, vertex1:Point2D, vertex2:Point2D, vertex3:Point2D, jointDepth:float):
        self.state = 'clean' #'clean' means unobstructed, 'lg' means obstructed by the lg 'motor' means obstructed by the motor, 'hinge' - by the wingtip hinge
        self.cellpts = [] #here be battery cells belonging to this triangle

        #adding an x dimension to a triangle
        self.xLE = None
        self.xTE = None
        self.cellxs = []

        self.v1 = vertex1
        self.v2 = vertex2
        self.v3 = vertex3
        self.maxy = max(self.v1.y, self.v2.y, self.v3.y) #useful for xwise packing

        #creating adhesion joint vertics - as in the April28 page
        self.v12 = UnitVector2D.from_pts(self.v1, self.v2).step(self.v1, jointDepth)
        self.v21 = UnitVector2D.from_pts(self.v2, self.v1).step(self.v2, jointDepth)
        self.v13 = UnitVector2D.from_pts(self.v1, self.v3).step(self.v1, jointDepth)
        self.v31 = UnitVector2D.from_pts(self.v3, self.v1).step(self.v3, jointDepth)
        self.v23 = UnitVector2D.from_pts(self.v2, self.v3).step(self.v2, jointDepth)
        self.v32 = UnitVector2D.from_pts(self.v3, self.v2).step(self.v3, jointDepth)

    #generating toothed wedges for the triangle
    def teeth(self, nteeth, toothDepth, tadh1, tadh2):
        # self.toothpts1 = self._tooth_generation(self.v1, self.toothpts1[0], self.toothpts1[1], nteeth, toothDepth)
        # self.toothpts2 = self._tooth_generation(self.v2, self.toothpts2[0], self.toothpts2[1], nteeth, toothDepth)
        # self.toothpts3 = self._tooth_generation(self.v3, self.toothpts3[0], self.toothpts3[1], nteeth, toothDepth)
        wedges.append(Wedge.toothed(self.v1, self.v12, self.v13, nteeth, toothDepth, tadh1, tadh2))
        wedges.append(Wedge.toothed(self.v2, self.v23, self.v21, nteeth, toothDepth, tadh1, tadh2))
        wedges.append(Wedge.toothed(self.v3, self.v31, self.v32, nteeth, toothDepth, tadh1, tadh2))

    #generating untoothed wedges for the triangle
    def no_teeth(self, tadh1, tadh2):
        wedges.append(Wedge.toothless(self.v1, self.v12, self.v13, tadh1, tadh2))
        wedges.append(Wedge.toothless(self.v2, self.v23, self.v21, tadh1, tadh2))
        wedges.append(Wedge.toothless(self.v3, self.v31, self.v32, tadh1, tadh2))

    def populate(self):
        self.cellpts = self._populate(self.v12, self.v13, self.v21, self.v23, self.v31, self.v32)

    #establishes the x bounds of the triangle and it's number of cell layers in x direction.
    def xbounds(self, lineLE:Line, lineTE:Line):       
        #caution! points inside this method are in yx coords instead of in yz coords, so z means x
        xLEpt = lineLE.intersect(Line(Point2D(self.maxy, 0, None), UnitVector2D(0, 1)))
        self.xLE = xLEpt.z
        xTEpt = lineTE.intersect(Line(Point2D(self.maxy, 0, None), UnitVector2D(0, 1)))
        if self.state == 'lg': #accounting for the shortening by lg. motor engines do not carry cells so it does not matter for them
            xTEpt = Point2D(xTEpt.y, xlg, None)
        self.xTE = xTEpt.z

        #non-battery carrying cells are accounted for by not having cellposes
        #packing batteries in x direction
        self.trimmed_xdist = self.xTE-self.xLE-delta #points are imported from catia, where xTE > xLE
        nbats = int(self.trimmed_xdist/(cell_h+delta))
        #there is no centering margin, for we want bats closer to TE so there is more space for stuff @LE
        descendvect = UnitVector2D(0, -1)
        batpos = descendvect.step(xTEpt, delta)
        for i in range(nbats):
            self.cellxs.append(batpos.z) #again, z means x
            batpos = descendvect.step(batpos, cell_h+delta) #moving on to the next cell

    def ncells(self): #once x bounds are defined, returns the total number of cells in the triangle
        return len(self.cellxs)*len(self.cellpts)


class Sheet():
    def __init__(self, s1, s2, s3, s4):
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4
        #to be assigned later
        self.xLE = None 
        self.xTE = None     
    

"""creation of the ray pattern"""
#uncoding the list - we need the 4 last points
codedPts = trapVertEnc.split('|')
fustop = [float(coord) for coord in codedPts[2].split(';')]
fusbot = [float(coord) for coord in codedPts[3].split(';')]
tiptop = [float(coord) for coord in codedPts[4].split(';')]
tipbot = [float(coord) for coord in codedPts[5].split(';')]

#calculating the angles to go up and down
dihedral = float(dihedral[:-3]) #in deg
up_angle = math.radians(60)+math.radians(dihedral)
down_angle = -math.radians(60)+math.radians(dihedral)

#claculating ines for top and bottom boundaries
riseTop = (fustop[2]-tiptop[2])/((fustop[1]-tiptop[1]))
interceptTop = fustop[2]-riseTop*fustop[1]

riseBot = (fusbot[2]-tipbot[2])/((fusbot[1]-tipbot[1]))
interceptBot = fusbot[2]-riseBot*fusbot[1]

#a submethod for solving linear equations 4 2 lines with correction for end of wing
def solveLines(alph, intercept, A, B):
    yproposed = (B-intercept)/(math.tan(alph)-A) #calculating a regular line intersect
    tooFar = yproposed >= tiptop[1] #checking whether it's outside of the wing
    yfinal = tiptop[1] if tooFar else yproposed #applying a correction to it, if necessary
    return yfinal, tooFar

#creating the diagonal ribs
intermediate2Dpoints = list()
if startTop: #starting from the top of the fuselage rect
    #collapse to 2d and add mandatory fuselage points
    intermediate2Dpoints.append(Point2D(fusbot[1], fusbot[2], False))
    current2Dpoint = Point2D(fustop[1], fustop[2], True) 
    intermediate2Dpoints.append(current2Dpoint)
    while True: #we continue until we break due to having crossed the entire wing
        #fist, we go downwards
        intercept = current2Dpoint.z-current2Dpoint.y*math.tan(down_angle) #we start with a line going down
        ynew, corrected = solveLines(down_angle, intercept, riseBot, interceptBot)
        current2Dpoint = Point2D(ynew, riseBot*ynew+interceptBot, False) #updating the point
        if corrected: #if it's a boundary pint for the wing we end the loop
            break
        intermediate2Dpoints.append(current2Dpoint) #adding the determined point

        #then we go upwards
        intercept = current2Dpoint.z-current2Dpoint.y*math.tan(up_angle) #now we use the up angle
        ynew, corrected = solveLines(up_angle, intercept, riseTop, interceptTop) #we will intersect with the top surface
        current2Dpoint = Point2D(ynew, riseTop*ynew+interceptTop, True) #updating the point
        if corrected: #if it's a boundary pint for the wing we end the loop
            break
        intermediate2Dpoints.append(current2Dpoint) #adding the determined point

else:
    #including mandatory fuselage points
    intermediate2Dpoints.append(Point2D(fustop[1], fustop[2], True))
    current2Dpoint = Point2D(fusbot[1], fusbot[2], False)
    intermediate2Dpoints.append(current2Dpoint)
    while True: #we continue until we break due to having crossed the entire wing
        #then we go upwards
        intercept = current2Dpoint.z-current2Dpoint.y*math.tan(up_angle) #now we use the up angle
        ynew, corrected = solveLines(up_angle, intercept, riseTop, interceptTop) #we will intersect with the top surface
        current2Dpoint = Point2D(ynew, riseTop*ynew+interceptTop, True) #updating the point
        if corrected: #if it's a boundary pint for the wing we end the loop
            break
        intermediate2Dpoints.append(current2Dpoint) #adding the determined point

        #fist, we go downwards
        intercept = current2Dpoint.z-current2Dpoint.y*math.tan(down_angle) #we start with a line going down
        ynew, corrected = solveLines(down_angle, intercept, riseBot, interceptBot)
        current2Dpoint = Point2D(ynew, riseBot*ynew+interceptBot, False) #updating the point
        if corrected: #if it's a boundary pint for the wing we end the loop
            break
        intermediate2Dpoints.append(current2Dpoint) #adding the determined point


#removing the last point if necessary, as including it might sometimes result in a very nasty geometry
if removeLast:
    del intermediate2Dpoints[-1]

#adding the tip points in the right order
if intermediate2Dpoints[-1].top:
    intermediate2Dpoints.append(Point2D(tipbot[1], tipbot[2], False))
    intermediate2Dpoints.append(Point2D(tiptop[1], tiptop[2], True))
else:
    intermediate2Dpoints.append(Point2D(tiptop[1], tiptop[2], True))
    intermediate2Dpoints.append(Point2D(tipbot[1], tipbot[2], False))

#setting up lists of x y z for point creation
xs = [fusbot[0]]*len(intermediate2Dpoints)
ys = [i.y for i in intermediate2Dpoints]
zs = [j.z for j in intermediate2Dpoints]

"""joint creation"""
#top and bottom direction vectors (according to apr 26 page of book of honours)
dirtop1 = UnitVector2D(fustop[1]-tiptop[1], fustop[2]-tiptop[2])
dirtop2 = UnitVector2D(tiptop[1]-fustop[1], tiptop[2]-fustop[2])
dirbot1 = UnitVector2D(fusbot[1]-tipbot[1], fusbot[2]-tipbot[2])
dirbot2 = UnitVector2D(tipbot[1]-fusbot[1], tipbot[2]-fusbot[2])
joints = list()

#fuselage end joint
dir2 = dirtop2 if intermediate2Dpoints[0].top else dirbot2
c0 = UnitVector2D.normal(dir2, not intermediate2Dpoints[0].top)
r2 = UnitVector2D.from_pts(intermediate2Dpoints[1], intermediate2Dpoints[0])
joints.append(Joint.edge2(intermediate2Dpoints[0], thicknesses[0], c0, dir2, r2))

#intermediate joints
for i in range(1, len(intermediate2Dpoints)-1):
    #using the appropriate direction vectors
    dir1 = dirtop1 if intermediate2Dpoints[i].top else dirbot1
    dir2 = dirtop2 if intermediate2Dpoints[i].top else dirbot2
    #the c vector, if we are at the top, c points down and vice versa
    cvec = UnitVector2D.normal(dir1, not intermediate2Dpoints[i].top)
    #r vectors
    r1 = UnitVector2D.from_pts(intermediate2Dpoints[i-1], intermediate2Dpoints[i])
    r2 = UnitVector2D.from_pts(intermediate2Dpoints[i+1], intermediate2Dpoints[i])

    #point generation
    joints.append(Joint.standard(intermediate2Dpoints[i], thicknesses[i-1], thicknesses[i], cvec, dir1, dir2, r1, r2))


#tip end joint
dir1 = dirtop1 if intermediate2Dpoints[-1].top else dirbot1
clast = UnitVector2D.normal(dir1, not intermediate2Dpoints[-1].top)
r1 = UnitVector2D.from_pts(intermediate2Dpoints[-1], intermediate2Dpoints[-2])
joints.append(Joint.edge1(intermediate2Dpoints[-1], thicknesses[i+1], clast, dir1, r1)) #i+1 since thicknesses have a different length

'''creation of sheet and triangle elements'''
#step 1. geometric outlines
triangles = list()
for i in range(1, len(intermediate2Dpoints)-1): #again without the edge point
    triangles.append(Triangle(joints[i-1].p2, joints[i].trigpt, joints[i+1].p1, jointDepth[i]))

sheets = list()
for i in range(0, len(intermediate2Dpoints)-1): #there is only 1 less sheet than joints so here the parametrization has to change
    sheets.append(Sheet(joints[i].trigpt, joints[i].p2, joints[i+1].trigpt, joints[i+1].p1))

#step 2. removing triangles that would coincide with the engines and the hinge, moving the triangles that overlap with lg to their own category
engineranges = [code.split(';') for code in motor_yzones.split('|')]
lgrange = lg_yzone.split(';')
#convert to float
for i in range(len(engineranges)):
    engineranges[i] = [float(engineranges[i][0]), float(engineranges[i][1])]
lgrange = [float(y) for y in lgrange]

for t in triangles:
    #check if not inside the motor
    for m in engineranges:
        if t.v1.yin(m) or t.v2.yin(m) or t.v3.yin(m):
            t.state = 'motor'
            break
    if t.state == 'clean': #the 'motor' state is most restrictive and hence takes priority
        if t.v1.yin(lgrange) or t.v2.yin(lgrange) or t.v3.yin(lgrange):
            t.state = 'lg'
#the hinge triangle
triangles[-1].state = 'hinge' #in practice same as motor, but just to be clear

#step 3. creating battery teeth for triangles not behind motors nor housing the hinge
for i, t in enumerate(triangles):
    if t.state == 'clean' or t.state == 'lg':
        t.teeth(nteeth[i], toothDepth[i], adhesive_t[i], adhesive_t[i+1])

        #step 4. populating the triangles that have to be populated
        t.populate()
    else: #generating tootless wedges for the rest
        t.no_teeth(adhesive_t[i], adhesive_t[i+1])

'''Mid-fuselage structure'''
#creating battery rectangle boundary points
roottop = Point2D(float(codedPts[0].split(';')[1]), float(codedPts[0].split(';')[2]), True)
rootbot = Point2D(float(codedPts[1].split(';')[1]), float(codedPts[1].split(';')[2]), False)
fusbotpt = Point2D(fusbot[1], fusbot[2], False)
fustoppt = Point2D(fustop[1], fustop[2], True)
roottopboundary = dirtop2.step(roottop, rect_joint_t+root_rib_offset)
fusbotboundary = dirbot1.step(fusbotpt, rect_joint_t+fus_skin_offset)
normaltop = UnitVector2D.normal(dirtop2, True)
#projection onto the top side near the fuselage boundary
fustopboundary = Line(fusbotboundary, normaltop).intersect(Line.from_pts(roottop, fustoppt))
#the 4th point is closing of the rectangle
rootbotboundary = Line(roottopboundary, normaltop).intersect(Line(fusbotboundary, dirtop2))

rootbatsides = [rootbotboundary, fusbotboundary, fustopboundary, roottopboundary]

#packing batteries
centerline = list() #creating centerline, starting from root bottom point
centerline.append(normaltop.step(dirtop2.step(rootbotboundary, delta+cell_d/2), delta+cell_d/2))
centerline.append(normaltop.step(dirtop2.step(fusbotboundary, -delta-cell_d/2), delta+cell_d/2))
centerline.append(normaltop.step(dirtop2.step(fustopboundary, -delta-cell_d/2), -delta-cell_d/2))
centerline.append(normaltop.step(dirtop2.step(roottopboundary, delta+cell_d/2), -delta-cell_d/2))

appended_centerline = centerline+[centerline[0]]
centroid_root = Point2D(sum([pt.y for pt in centerline])/4, sum([pt.z for pt in centerline])/4, None)
root_nvs = list()
root_lines = list()
sidevect = UnitVector2D.from_pts(appended_centerline[0], appended_centerline[0+1])
minusdir = sidevect.rotate(120, True)
plusdir = sidevect.rotate(60, True)
for i in range(len(centerline)):
    root_nvs.append(UnitVector2D.normal_towards(UnitVector2D.from_pts(appended_centerline[i], appended_centerline[i+1]), UnitVector2D.from_pts(appended_centerline[i], centroid_root)))
    root_lines.append(Line.from_pts(appended_centerline[i], appended_centerline[i+1]))

cellpts = list()
cellptsrow = list() #here be cells from the current row, which after checking for inside/outside will be added to correctedRow
correctedRow = list() #after branching into other points will be added to cellpts

width1strow=appended_centerline[0].dist(appended_centerline[1])
ncells1stRow = 1+int(width1strow/delta_diameter)
integer_margin = (width1strow-(ncells1stRow-1)*delta_diameter)/2 #to center the cell pattern
for i in range(ncells1stRow):
    correctedRow.append(sidevect.step(appended_centerline[0], integer_margin+i*delta_diameter)) #here we know that this point is correct

#cell creation - remaining layers
while correctedRow != []:
#for i in range(20): #4 tests
    cellptsrow.append(minusdir.step(correctedRow[0], delta_diameter)) #generating the one point that can only be created by going in the - dir
    for pt in correctedRow: #the remaining points can be created by going in the plus direction
        cellptsrow.append(plusdir.step(pt, delta_diameter))            
    #moving the correctedRow to cellpts once their job is done
    cellpts += correctedRow
    correctedRow = list()
    #checking if the generated points are inside
    for pt in cellptsrow:
        inside = True
        for index, edge in enumerate(root_lines):
            if not edge.sidecheck(pt, root_nvs[index]):
                inside = False
                #break
        if inside: #if the point is inside, it gets added to the next propagation list
            correctedRow.append(pt)
    cellptsrow = list() #clearing the row list


'''Computing the between spar distance for each triangle'''
#decoding input
angleLE = float(fsang[:-3])
angleTE =float(rsang[:-3])
LEref = LEenc.split(";")
TEref = TEenc.split(";")
yref = float(LEref[1])
ydir = UnitVector2D(1, 0)

#!ATTENTION! THE BELOW VECTORS ARE IN YX INSTEAD OF YZ SO POINT.Z REPRESENTS THE X COORD IN THE CAD!!!
xLEptref = Point2D(yref, float(LEref[0]), None)
xTEptref = Point2D(yref, float(TEref[0]), None)
LEline = Line(xLEptref, ydir.rotate(angleLE, True))
TELine = Line(xTEptref, ydir.rotate(angleTE, False))

#packing and counting batteries
cellcount = 0
for t in triangles:
    t.xbounds(LEline, TELine)
    cellcount+=t.ncells()

#the in-fuselage section
fuscellxs = []
towardsLE = UnitVector2D(0,-1)
fusLETELine = Line(Point2D(fusbotpt.y, 0, None), towardsLE)
fusLEpt = LEline.intersect(fusLETELine)
fusTEpt = TELine.intersect(fusLETELine)
fus_xdist = fusTEpt.z-fusLEpt.z-delta #xTE always > xLE! Account for the +1th delta
nbatsFus = int(fus_xdist/(cell_h+delta))
centering_margin = (fus_xdist-nbatsFus*(cell_h+delta))/2
currentpt = towardsLE.step(fusTEpt, delta+centering_margin)
for i in range(nbatsFus):
    fuscellxs.append(currentpt.z)
    currentpt = towardsLE.step(currentpt, delta+cell_h)
cellcount+=len(fuscellxs)*len(cellpts)

for i, t in enumerate(triangles):
    wedges[3*i].xLE = t.xLE
    wedges[3*i].xTE = t.xTE
    wedges[3*i+1].xLE = t.xLE
    wedges[3*i+1].xTE = t.xTE
    wedges[3*i+2].xLE = t.xLE
    wedges[3*i+2].xTE = t.xTE
    sheets[i].xLE = t.xLE
    sheets[i].xTE = t.xTE

#adjusting x positions of edge sheets
sheets[0].xLE = fusLEpt.z
sheets[0].xTE = fusTEpt.z
finalsheetLine = Line(Point2D(max(sheets[-1].s1.y, sheets[-1].s2.y, sheets[-1].s3.y, sheets[-1].s4.y), 0, None), towardsLE)
sheets[-1].xLE = finalsheetLine.intersect(LEline).z
sheets[-1].xTE = finalsheetLine.intersect(TELine).z

#adjusting the edge of joints


'''!Output Lists!'''
#everything is converted to meters as that is what is projected into ekl objects
#sheets
s1ys = [s.s1.y/1000 for s in sheets]
s1zs = [s.s1.z/1000 for s in sheets]
s2ys = [s.s2.y/1000 for s in sheets]
s2zs = [s.s2.z/1000 for s in sheets]
s3ys = [s.s3.y/1000 for s in sheets]
s3zs = [s.s3.z/1000 for s in sheets]
s4ys = [s.s4.y/1000 for s in sheets]
s4zs = [s.s4.z/1000 for s in sheets]
sLEs = [s.xLE/1000 for s in sheets]
sTEs = [s.xTE/1000 for s in sheets]

#cells - for now all in one list set
cellxs, cellys, cellzs = list(), list(), list()
for pt in cellpts: #first the fuselage cells
    for x in fuscellxs:
        cellxs.append(x/1000)
        cellys.append(pt.y/1000)
        cellzs.append(pt.z/1000)
for t in triangles:
    for x in t.cellxs:
        for pt in t.cellpts:
            cellxs.append(x/1000)
            cellys.append(pt.y/1000)
            cellzs.append(pt.z/1000)

#wedges - again one list set
wmainys = [w.main_vertex.y/1000 for w in wedges]
wmainzs = [w.main_vertex.z/1000 for w in wedges]
wLEs = [w.xLE/1000 for w in wedges]
wTEs = [w.xTE/1000 for w in wedges]
wtoothys = [[pt.y/1000 for pt in w.toothpts] for w in wedges]
wtoothzs = [[pt.z/1000 for pt in w.toothpts] for w in wedges]

#triangles - again one list set, will require index calibration to get the wavy pattern from wedges
v12ys = [t.v12.y/1000 for t in triangles]
v12zs = [t.v12.z/1000 for t in triangles]
v13ys = [t.v13.y/1000 for t in triangles]
v13zs = [t.v13.z/1000 for t in triangles]
v23ys = [t.v23.y/1000 for t in triangles]
v23zs = [t.v23.z/1000 for t in triangles]
v21ys = [t.v21.y/1000 for t in triangles]
v21zs = [t.v21.z/1000 for t in triangles]
v32ys = [t.v32.y/1000 for t in triangles]
v32zs = [t.v32.z/1000 for t in triangles]
v31ys = [t.v31.y/1000 for t in triangles]
v31zs = [t.v31.z/1000 for t in triangles]
tLEs = [t.xLE/1000 for t in triangles]
tTEs = [t.xTE/1000 for t in triangles]

'''END COPY-PASTE HERE'''
#trapezoid outline
plt.subplot(2, 1, 1)
plt.title(f"Front view of the 'wingbox', {cellcount} cells in total")
plt.xlabel("spanwise position [mm]", loc="right")
plt.ylabel("height [mm]")
plt.plot([fustop[1], fusbot[1], tipbot[1], tiptop[1], fustop[1]], [fustop[2], fusbot[2], tipbot[2], tiptop[2], fustop[2]])
plt.plot(ys, zs)
for j in joints:
    plt.plot([j.p1.y, j.p2.y, j.trigpt.y, j.p1.y], [j.p1.z, j.p2.z, j.trigpt.z, j.p1.z])
for t in triangles:
    plt.plot([t.v1.y, t.v2.y, t.v3.y, t.v1.y], [t.v1.z, t.v2.z, t.v3.z, t.v1.z])
    plt.scatter([pt.y for pt in t.cellpts], [pt.z for pt in t.cellpts])
for s in sheets:
    plt.plot([s.s1.y, s.s2.y, s.s3.y, s.s4.y, s.s1.y], [s.s1.z, s.s2.z, s.s3.z, s.s4.z, s.s1.z])
for w in wedges:
    plt.plot([w.main_vertex.y]+[pt.y for pt in w.toothpts]+[w.main_vertex.y],
             [w.main_vertex.z]+[pt.z for pt in w.toothpts]+[w.main_vertex.z])

#inner trapezoid outline
plt.plot([fusbotpt.y, fustoppt.y, roottop.y, rootbot.y, fusbotpt.y], [fusbotpt.z, fustoppt.z, roottop.z, rootbot.z, fusbotpt.z])
plt.plot([pt.y for pt in rootbatsides+[rootbatsides[0]]], [pt.z for pt in rootbatsides+[rootbatsides[0]]])
plt.plot([pt.y for pt in appended_centerline], [pt.z for pt in appended_centerline])
plt.scatter([pt.y for pt in cellpts], [pt.z for pt in cellpts])

#top view
plt.subplot(2, 1, 2)
plt.title("Top view of the 'wingbox'")
plt.xlabel("spanwise position [mm]", loc="right")
plt.ylabel("chordwise coord. [mm]")
for t in triangles:
    plt.plot([t.maxy, t.maxy], [t.xTE, t.xLE])
    cellys, cellxs = [], []
    for x in t.cellxs:
        for pt in t.cellpts:
            cellys.append(pt.y), cellxs.append(x)
    plt.scatter(cellys, cellxs)
#fuselage batteries
fusbatxs = []
fusbatys = []
for pt in cellpts:
    for x in fuscellxs:
        fusbatxs.append(x)
        fusbatys.append(pt.y)
plt.scatter(fusbatys, fusbatxs)
for s in sheets:
    plt.plot([s.s1.y, s.s1.y], [s.xLE, s.xTE])
for w in wedges:
    plt.plot([w.main_vertex.y, w.main_vertex.y], [w.xLE, w.xTE])

# #how each triangle contributes to cell count
# plt.subplot(3,1,3)
# plt.title(f"distribution of {cellcount} cells over the wing")
# plt.xlabel("spanwise position [mm]", loc="right")
# plt.ylabel("amount of cells per primatic pack [-]")
# ys = [fusbotpt.y]
# hs = [len(fuscellxs)*len(cellpts)]
# for t in triangles:
#     ys.append(t.maxy)
#     hs.append(t.ncells())
# plt.plot(ys, hs)

plt.show()
