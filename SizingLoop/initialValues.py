import componentClasses as ccl
import angledRibPattern as arp
import typing as ty

'''generating initial values for sizing loop based on created rib geometry and assumptions'''
def initial_components(joints:ty.List[arp.JointPoints])->ty.List[ccl.Component]:
    '''create the spar, skins, ribs, rivets and lump masses of the wing'''
    raise NotImplementedError