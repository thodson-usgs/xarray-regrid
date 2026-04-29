from typing import TYPE_CHECKING

import numpy as np

SHAPELY_IMPORT_ERROR = (
    "polygon conservative regridding requires shapely >= 2.0; "
    "install with `pip install shapely`."
)

try:
    import shapely
    from shapely import affinity
    from shapely.strtree import STRtree

    HAS_SHAPELY = True
except ImportError:  # pragma: no cover
    shapely = None
    affinity = None
    STRtree = None
    HAS_SHAPELY = False

try:
    import sparse

    HAS_SPARSE = True
except ImportError:  # pragma: no cover
    sparse = None
    HAS_SPARSE = False

if TYPE_CHECKING:
    import sparse as sparse_mod

    AreaMatrix = sparse_mod.COO | np.ndarray
else:
    AreaMatrix = np.ndarray


def require_shapely() -> None:
    if not HAS_SHAPELY:
        raise ImportError(SHAPELY_IMPORT_ERROR)
