import typing as ty
import aerosandbox.numpy as np
import numpy.typing as nt

class Mesher:
    def __init__(self, decimalPlaces:int=5):
        self.nodes:ty.List[ty.Tuple[float]] = list() #tuples in format (x, y, z)
        
        self.eleTypes:ty.List[str] = list()
        self.eleNodePoses:ty.List[ty.List[int]] = list()
        self.eleArgs:ty.List[ty.List[float]] = list()

        self._nodesHashMap:ty.Dict[ty.Tuple[int], int] = dict()
        self._DECIMAL_PLACES = decimalPlaces
        self._submeshIdxHashMap:ty.Dict[str, ty.Dict[int,object]] = dict()

    def load_ele(self, eleNodes:ty.List[ty.Tuple[float]], eleType:str, eleArg:ty.List[float]):
        '''
        Adds an element to the mesh
        
        :param eleNodes: list of coordinates in 3-space of the element node points
        :type eleNodes: ty.List[ty.Tuple]
        :param eleType: a string id corresponding to the in-model element types, i.e. a skin panel - 'sk'
        :type eleType: str
        :param eleArg: arguments needed to initialise an element of a given eleType
        :type eleArg: ty.List[float]
        '''
        #0) passing forth the eleType and eleArg
        self.eleTypes.append(eleType)
        self.eleArgs.append(eleArg)
        if not eleType in self._submeshIdxHashMap: #extending the submeshes if a new element type is encountered 
            self._submeshIdxHashMap[eleType] = dict()

        #1) preparing hash entries for the element coordinates
        to_tuple_entry = lambda _float:int(round(_float*10**self._DECIMAL_PLACES))
        hashEntries = [(to_tuple_entry(eleNode[0]), to_tuple_entry(eleNode[1]), to_tuple_entry(eleNode[2])) for eleNode in eleNodes]

        #2) creating a set of elemnt node ids corresponding to hash entries, adding the original hash entries to the hash dict and node ids
        eleNodePos:ty.List[int] = list()
        for hashEntry, eleNode in zip(hashEntries, eleNodes):
            if not(hashEntry in self._nodesHashMap):
                self._nodesHashMap[hashEntry] = len(self.nodes) #'ll point to the index of the new node
                self.nodes.append(eleNode)
            nodePos = self._nodesHashMap[hashEntry]
            eleNodePos.append(nodePos)

            #3) adding the processed node to relevant submeshes
            if not(nodePos in self._submeshIdxHashMap[eleType]):
                self._submeshIdxHashMap[eleType][nodePos] = None #we only need hashes of the ids for repetition lookup, the value can be left empty

        self.eleNodePoses.append(eleNodePos)

    def get_submesh(self, eleType:str) -> ty.Tuple[nt.NDArray[np.int32], nt.NDArray[np.float64]]:
        '''
        Returns node indices and node coordinates in the pyfe3d ncoords format that are connected to at least one element of the provided eleType
        
        :param eleType: the string id of the element type for which we want to obtain the corrected nodes
        :type eleType: str
        :return: Description
        :rtype: Tuple[NDArray[int32], NDArray[float64]]
        '''
        indices = np.array(list(self._submeshIdxHashMap[eleType].keys()), dtype=np.int32)
        return indices, np.array(self.nodes)[indices, :]
        