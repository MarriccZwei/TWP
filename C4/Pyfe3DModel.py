import pyfe3d as pf3
import numpy as np
import numpy.typing as nt
import typing as ty
import pyfe3d.shellprop as psp

class Pyfe3DModel():
    def __init__(self, ncoords:nt.NDArray[np.float32]):
        assert ncoords.shape[1]==3
        self.ncoords = ncoords
        self.x = ncoords[:,0]
        self.y = ncoords[:,1]
        self.z = ncoords[:,2]
        self.ncoords_flatten = ncoords.flatten()

        self.beamprobe = pf3.BeamCProbe()
        self.qad_probe = pf3.Quad4Probe()

        self.beams:ty.List[pf3.BeamC] = list()
        self.beamprops:ty.List[pf3.BeamProp] = list()
        self.beamrots:ty.List[ty.Tuple[int]] = list()

        self.quads:ty.List[pf3.Quad4] = list()
        self.shellprops:ty.List[psp.ShellProp] = list()

    def add_quad(self, pos1, pos2, pos3, pos4):
        pass

    def add_beam(self, pos1, pos2):
        pass

    def KC0_update(self, beamprops, beamorients, shellprops):
        pass