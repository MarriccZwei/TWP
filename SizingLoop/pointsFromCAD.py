import geometricClasses as gcl
import typing as ty

#TODO: decode the necessary points from catia

'''Corner Points'''
rft:gcl.Point3D = NotImplemented #root-front-top, others named analogically
rfb:gcl.Point3D = NotImplemented
rrt:gcl.Point3D = NotImplemented
rrb:gcl.Point3D = NotImplemented
fft:gcl.Point3D = NotImplemented
ffb:gcl.Point3D = NotImplemented
frt:gcl.Point3D = NotImplemented
frb:gcl.Point3D = NotImplemented
tft:gcl.Point3D = NotImplemented
tfb:gcl.Point3D = NotImplemented
trt:gcl.Point3D = NotImplemented
trb:gcl.Point3D = NotImplemented

'''Engine Locations'''
engineStarts:ty.List[float] = NotImplemented #spanwise y coordinates
engineEnds:ty.List[float] = NotImplemented

'''Landing Gear Locations'''
lgStart:gcl.Point2D = NotImplemented #x@landing gear end, y@spanwise position
lgEnd:gcl.Point2D = NotImplemented
