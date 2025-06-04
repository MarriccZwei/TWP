import geometricClasses as gcl
import typing as ty

#TODO: decode the necessary points from catia

class PointsFromCAD():
    def __init__(self, corners:ty.Dict[str, gcl.Point3D], engineR:float, engineCgs:ty.List[gcl.Point3D], lgStart:gcl.Point2D, lgEnd:gcl.Point2D, lgCg:gcl.Point3D):
        '''Corner Points'''
        self.rft:gcl.Point3D = corners["rft"] #root-front-top, others named analogically
        self.rfb:gcl.Point3D = corners["rfb"]
        self.rrt:gcl.Point3D = corners["rrt"]
        self.rrb:gcl.Point3D = corners["rrb"]
        self.fft:gcl.Point3D = corners["fft"]
        self.ffb:gcl.Point3D = corners["ffb"]
        self.frt:gcl.Point3D = corners["frt"]
        self.frb:gcl.Point3D = corners["frb"]
        self.tft:gcl.Point3D = corners["tft"]
        self.tfb:gcl.Point3D = corners["tfb"]
        self.trt:gcl.Point3D = corners["trt"]
        self.trb:gcl.Point3D = corners["trb"]

        '''Engine Locations'''
        self.engineStarts:ty.List[float] = [cg.y-engineR for cg in engineCgs] #spanwise y coordinates
        self.engineEnds:ty.List[float] = [cg.y+engineR for cg in engineCgs]
        self.engineCgs = engineCgs

        '''Landing Gear Locations'''
        self.lgStart:gcl.Point2D = lgStart #x@landing gear end, y@spanwise position
        self.lgEnd:gcl.Point2D = lgEnd
        self.lgCg = lgCg

    @classmethod
    def testpoints(cls):
        return cls({'rft':gcl.Point3D(0, 0, 0.25), 'rfb':gcl.Point3D(0, 0, -0.25), 'rrt':gcl.Point3D(1, 0, 0.25), 'rrb':gcl.Point3D(1, 0, -0.25),
                    'fft':gcl.Point3D(0.1, 1.2, 0.15), 'ffb':gcl.Point3D(0.1, 1.2, -0.2), 'frt':gcl.Point3D(0.9, 1.2, 0.15), 'frb':gcl.Point3D(0.9, 1.2, -.2),
                    'tft':gcl.Point3D(0.2, 10.2, 0), 'tfb':gcl.Point3D(0.2, 10.2, -0.15), 'trt':gcl.Point3D(0.6, 10.2, 0.15), 'trb':gcl.Point3D(0.6, 10.2, -.15)},
                    .1,[gcl.Point3D(-.4, 3.1, 0), gcl.Point3D(0, 7.1, 0)], gcl.Point2D(0.5, 4), gcl.Point2D(0.5, 4.35), gcl.Point3D(1.1, 4.175, -.1))
    
    @classmethod
    def decode(cls):
        raise NotImplementedError
