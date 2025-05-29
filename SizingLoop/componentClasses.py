import abc
import angledRibPattern as arp
import geometricClasses as gcl
import typing as ty

class Component(abc.ABC):
    '''In the init, it should have arguments allowing it to obtain global properties of the component,
    such us inertia params per unit length/width in a given direction'''

    @abc.abstractmethod
    def shard(self, nodes:ty.List[gcl.Point3D]):
        '''returns an element connected to the specfied nodes, with properties corresponding of the global elements of the component'''
        raise NotImplementedError
    