import typing as ty

class CatiaParser:
    def __init__(self, cadoutstr:str):
        """
        Reads the mesh from the catia export string
        
        :param cadoutstr: catia export string, formatted as elementTypeCode;arg1,arg2,etc.;node1x$node1y$node1z,node2x$node2y$node2z,etc.|
        :type cadoutstr: str
        """
        self._eleTypes:ty.List[str] = list()
        self._eleArgs:ty.List[ty.List[float]] = list()
        self._eleNodes:ty.List[ty.List[ty.Tuple[float]]] = list()
        try:
            for eleDataStr in cadoutstr.split("|"):
                eleType, eleArgsStr, eleNodesStr = eleDataStr.split(";")

                #2.1) element type
                self._eleTypes.append(eleType)
                
                #2.2) element args
                eleArgs = [] #might be this list stays empty should no args be needed
                if len(eleArgsStr)>0:
                    for eleArgStr in eleArgsStr.split(","):
                        eleArgs.append(float(eleArgStr))
                self._eleArgs.append(eleArgs)

                #2.3) element node coordinates
                eleNodes = []
                for eleNodeStr in eleNodesStr.split(","):
                    eleNodeXStr, eleNodeYStr, eleNodeZStr = eleNodeStr.split("$")
                    eleNodes.append((float(eleNodeXStr), float(eleNodeYStr), float(eleNodeZStr)))
                self._eleNodes.append(eleNodes)
        except ValueError:
            raise ValueError(f"Mis-formatted export string: {eleDataStr}")


    def get_mesh_data(self):
        return zip(self._eleTypes, self._eleArgs, self._eleNodes)
