import aerosandbox.numpy as np
import numpy.typing as nt
from collections import defaultdict


class Exclusion:

    def __init__(
        self,
        scaffold: nt.NDArray[np.float64],
        rjc: float,
        c_at_y,
    ):
        """
        Excludes points that lie within a radius of scaffold points.

        Radius for scaffold point with coordinate y:
            r = rjc * c_at_y(y)

        c_at_y does NOT support batch evaluation.
        """

        pts = np.reshape(scaffold, (-1, 3))
        self.points = pts

        # compute radii (cannot vectorize because c_at_y isn't batchable)
        radii = []
        for p in pts:
            radii.append(rjc * c_at_y(p[1]))
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

                        if np.sum((point - sp)**2) <= r*r:
                            return True

        return False