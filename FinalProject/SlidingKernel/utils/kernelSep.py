# Define the utils so that I might give you a dictionary of atoms positions, and you will split it into various kernels
import numpy as np

import numpy as np
from typing import List, Dict, Generator, Tuple

def iterate_kernels(
    atoms: List[Dict],
    bounds: np.ndarray,
    kernel_size: Tuple[float, float, float],
    stride: Tuple[float, float, float] | None = None,
) -> Generator[Dict, None, None]:
    """
    Yields one kernel at a time, each a dict:
      {
        'center': (cx, cy, cz),         # kernel centre
        'box_size': (dx, dy, dz),       # kernel extents (for later PBC use)
        'atoms':   [...]                # *shallow copies* of atom dicts inside
      }

    Notes
    -----
    • Periodic boundary conditions are enforced with the minimum-image convention.  
    • If `stride` is None, the kernels are non-overlapping (stride == kernel_size).  
    • Designed for large trajectories – keeps only one kernel in memory at once.
    """
    kernel_size = np.asarray(kernel_size, float)
    if stride is None:
        stride = kernel_size
    stride      = np.asarray(stride, float)

    box_len   = bounds[:, 1] - bounds[:, 0]
    starts    = kernel_size / 2.0
    n_steps   = np.ceil((box_len - kernel_size) / stride).astype(int) + 1

    pos = np.array([[a["x"], a["y"], a["z"]] for a in atoms])

    for ix in range(n_steps[0]):
        cx = starts[0] + ix * stride[0]
        for iy in range(n_steps[1]):
            cy = starts[1] + iy * stride[1]
            for iz in range(n_steps[2]):
                cz = starts[2] + iz * stride[2]
                centre = np.array([cx, cy, cz])

                disp   = np.abs(pos - centre)
                disp   = np.where(disp > box_len / 2.0, box_len - disp, disp)
                in_box = np.all(disp <= kernel_size / 2.0, axis=1)

                yield {
                    "center":   centre,
                    "box_size": kernel_size,
                    "atoms":    [atoms[i] for i in np.where(in_box)[0]],
                }

def split_atoms_by_kernel(atoms: List[Dict], kernel_size, stride=None, bounds=None):
    """
    Wrapper that *returns* a generator instead of pre-allocating all kernels.

    Example
    -------
    >>> bounds, atoms = read_lammps_dump('dump.lammpstrj')
    >>> for ker in split_atoms_by_kernel(atoms, (5, 5, 5), stride=(2.5, 2.5, 2.5), bounds=bounds):
    ...     do_my_computation(ker)           # ker['atoms'] is just this window
    """
    if bounds is None:
        raise ValueError("`bounds` must be supplied (output of read_lammps_dump).")
    return iterate_kernels(atoms, bounds, kernel_size, stride)