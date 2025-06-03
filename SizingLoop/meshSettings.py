class Meshsettings():
    def __init__(self, nb:int, nc:int, nbf:int):
        self.nb = nb #spanwise element count
        self.nc = nc #chordwise element count
        #chordwise count shall be uniform st same nodes can be reused
        self.nbf = nbf #spanwise element count per component for flanges
    
    @classmethod
    def default(cls): #the mesh settings used in main
        raise NotImplementedError