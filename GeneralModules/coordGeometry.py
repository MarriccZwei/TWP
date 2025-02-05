import numpy as np
#some 3D coordinate geometry stuff
def centroid(vertices):
   return (np.average(np.array([vertices[i][0] for i in range(len(vertices))])),
                            np.average(np.array([vertices[i][1] for i in range(len(vertices))])),
                            np.average(np.array([vertices[i][2] for i in range(len(vertices))])),)

def normalThroughPoint(pt, pt1, pt2):
   pass