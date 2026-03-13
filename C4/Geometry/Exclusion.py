import aerosandbox.numpy as np
import numpy.typing as nt
from collections import defaultdict


class Exclusion:

    def __init__(
        self,
        scaffold: nt.NDArray[np.float64],
        rjc: float,
        c_at_y,
        epsilon = .1
    ):
        """
        Excludes points that lie within a radius of scaffold points.

        Radius for scaffold point with coordinate y:
            r = rjc * c_at_y(y)

        c_at_y does NOT support batch evaluation.
        """

        pts = np.reshape(scaffold, (-1, 3))
        self.points = pts

        self.epsilon = epsilon
        self.c_at_y = c_at_y
        self.rjc = rjc

        # compute radii (cannot vectorize because c_at_y isn't batchable)
        radii = []
        for p in pts:
            radii.append(rjc * c_at_y(p[1]) * (1-epsilon))
        self.radii = np.array(radii)

        self.max_r = float(np.max(self.radii))

        # grid size = max radius
        self.cell_size = self.max_r

        self.grid = defaultdict(list)

        for i, p in enumerate(pts):
            cell = self._cell_index(p)
            self.grid[cell].append(i)

    def _cell_index(self, p):
        return tuple(np.floor(p / self.cell_size).astype(int))

    def is_excluded(self, point: nt.NDArray[np.float64]) -> bool:

        base = self._cell_index(point)

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):

                    cell = (base[0]+dx, base[1]+dy, base[2]+dz)

                    if cell not in self.grid:
                        continue

                    for idx in self.grid[cell]:

                        sp = self.points[idx]
                        r = self.radii[idx]

                        if abs(point[1]-sp[1]) <= r:
                            local_r = (1-self.epsilon)*self.c_at_y(point[1])*self.rjc
                            if (point[0]-sp[0])**2+(point[2]-sp[2])**2 <= local_r**2:
                                return True

        return False