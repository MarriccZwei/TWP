import abc
from ..Pyfe3DModel import Pyfe3DModel
from ..Config.Config import Config
import numpy as np
import numpy.typing as nt
import typing as ty

class AdditionalMeshOutputs():
    def __init__(self,
             skinTopCoords: nt.ArrayLike, #these arrays will be filled with gcl.Point3D objects, not typable in nt
             skinBotCoords: nt.ArrayLike,
             skinTopIds: nt.NDArray[np.int32],
             skinBotIds: nt.NDArray[np.int32],
             lgIds: nt.NDArray[np.int32],
             motorIds: ty.List[ty.List[np.int32]]):

        self.skinTopCoords = skinTopCoords
        self.skinBotCoords = skinBotCoords
        self.skinTopIds = np.asarray(skinTopIds, dtype=np.int32)
        self.skinBotIds = np.asarray(skinBotIds, dtype=np.int32)
        self.lgIds = np.asarray(lgIds, dtype=np.int32)
        self.motorIds = motorIds

        #mesh element specifications
        self.quadEleIds:ty.List[str] = list()
        self.quadArgs:ty.List = list()
        self.beamEleIds:ty.List[str] = list()
        self.beamArgs:ty.List = list()
        self.inertiaVals:ty.List[ty.Tuple[int]] = list()


    def load_quad(self, eleid:str, args):
        self.quadEleIds.append(eleid)
        self.quadArgs.append(args)

    def load_beam(self, eleid:str, args):
        self.beamEleIds.append(eleid)
        self.beamArgs.append(args)

    def load_inertia(self, inertiaM, inertiaJxx, inertiaJyy, inertiaJzz):
        self.inertiaVals.append((inertiaM, inertiaJxx, inertiaJyy, inertiaJzz))



class ModelInitInterface(abc.ABC):
    @abc.abstractmethod
    def modelInit(self, config: Config) -> ty.Tuple[Pyfe3DModel, AdditionalMeshOutputs]:
        '''Given a config file, will initialise the a Pyfe3DModel'''
        raise NotImplementedError
    
