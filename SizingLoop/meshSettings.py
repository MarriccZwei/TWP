class Meshsettings():
    def __init__(self, bsize, csize, flangeBsize, flangeCsize):
        self.bsize = bsize
        self.csize = csize
        self.flangeBsize = flangeBsize
        self.flangeCsize = flangeCsize
    
    @classmethod
    def default(cls): #the mesh settings used in main
        raise NotImplementedError