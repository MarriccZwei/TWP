import geometricClasses as gcl
import typing as ty
import numpy.typing as nt

'''Here be a file with meshings of given components. 
Inside-component connections will be completed in this file, 
some connections might have to be applied in main'''
def trigspars(mesh:gcl.Mesh3D, ntrig:int, rivet:str, sheet:str, 
              ffb:gcl.Point3D, frb:gcl.Point3D, frt:gcl.Point3D, 
              fft:gcl.Point3D,tfb:gcl.Point3D, trb:gcl.Point3D, 
              trt:gcl.Point3D, tft:gcl.Point3D)->ty.List[nt.ArrayLike]:
    def trig_crossec(fb:gcl.Point3D, rb:gcl.Point3D, rt:gcl.Point3D, 
                     ft:gcl.Point3D)->ty.List[ty.List[gcl.Point3D]]:
        a = fb.pythagoras(ft) #the side of the battery triangle
        f = ft.pythagoras(rt)/ntrig-a #the full flange width
        
        #first sheet, the inwards curved

