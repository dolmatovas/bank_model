from __future__ import annotations

try:
    from .interpolation import (
        build_interp3_stencil,
        clip_numba,
        interp3_from_stencil,
        interp3_from_stencil_numba,
        interp3_uniform_numpy,
        njit,
    )
except ImportError:
    from interpolation import (  # type: ignore
        build_interp3_stencil,
        clip_numba,
        interp3_from_stencil,
        interp3_from_stencil_numba,
        interp3_uniform_numpy,
        njit,
    )


clip = clip_numba
interp3_uniform = interp3_uniform_numpy
