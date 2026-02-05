import numpy.typing as nt
import numpy as np
import typing as ty

def adaptive_mesh_wall(ip1:nt.NDArray[np.float64], ip2:nt.NDArray[np.float64], 
                       op1:nt.NDArray[np.float64], op2:nt.NDArray[np.float64],
                       ys:nt.NDArray[np.float64], ni:int, no:int, 
                       eleArgFun:ty.Callable[[nt.NDArray[np.float64]], list])->ty.Tuple[
                           ty.List[ty.List[ty.Tuple[float]]], ty.List[ty.List[float]]]:
    '''
    Docstring for adaptive_mesh_wall
    
    :param ip1: inboard point with lower x
    :type ip1: nt.NDArray[np.float64]
    :param ip2: inboard point with higher x
    :type ip2: nt.NDArray[np.float64]
    :param op1: outboard point with lower x
    :type op1: nt.NDArray[np.float64]
    :param op2: outboard point with higher x
    :type op2: nt.NDArray[np.float64]
    :param ys: y values at which the nodes are to be created (rhombing excluded), 
    starting inboard
    :type ys: nt.NDArray[np.float64]
    :param ni: number of inboard nodes, has to be odd
    :type ni: int
    :param no: number of outboard nodes, has to be odd
    :type no: int
    :param eleArgFun: a function to assign eleArgs to an element based on its coordinates
    in an ncoords format
    :type eleArgFun: ty.Callable[[nt.NDArray[np.float64]], list]
    :return: the list of element coordinates for each created element and
    the list of element arguments for each created element
    :rtype: Tuple[List[List[Tuple[float]]], List[List[float]]]
    '''
    elesNodes:ty.List[ty.List[ty.Tuple[float]]] = list()
    elesArgs:ty.List[ty.List[float]] = list()

    assert ys[0]==ip1[1]==ip2[1]
    assert ys[-1]==op1[1]==op2[1]
    layerRhomb:list[bool] = list()
    layerRes:list[float] = list()
    refineInboard = ni>no

    return elesNodes, elesArgs
