from __future__ import annotations

import numpy as np
import numba as nb


njit = nb.njit


@njit(cache=True)
def clip_numba(x, lower, upper):
    return np.minimum(np.maximum(x, lower), upper)


def interp3_uniform_numpy(
    values: np.ndarray,
    x,
    y,
    z,
    r_grid: np.ndarray,
    q_grid: np.ndarray,
    k_grid: np.ndarray,
) -> np.ndarray:
    """Fast trilinear interpolation on a uniform 3D grid using NumPy."""
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    x = np.clip(x, r_grid[0], r_grid[-1])
    y = np.clip(y, q_grid[0], q_grid[-1])
    z = np.clip(z, k_grid[0], k_grid[-1])

    dx = r_grid[1] - r_grid[0]
    dy = q_grid[1] - q_grid[0]
    dz = k_grid[1] - k_grid[0]

    ix = np.floor((x - r_grid[0]) / dx).astype(np.int64)
    iy = np.floor((y - q_grid[0]) / dy).astype(np.int64)
    iz = np.floor((z - k_grid[0]) / dz).astype(np.int64)

    ix = np.clip(ix, 0, len(r_grid) - 2)
    iy = np.clip(iy, 0, len(q_grid) - 2)
    iz = np.clip(iz, 0, len(k_grid) - 2)

    tx = (x - r_grid[ix]) / dx
    ty = (y - q_grid[iy]) / dy
    tz = (z - k_grid[iz]) / dz

    c000 = values[ix, iy, iz]
    c001 = values[ix, iy, iz + 1]
    c010 = values[ix, iy + 1, iz]
    c011 = values[ix, iy + 1, iz + 1]
    c100 = values[ix + 1, iy, iz]
    c101 = values[ix + 1, iy, iz + 1]
    c110 = values[ix + 1, iy + 1, iz]
    c111 = values[ix + 1, iy + 1, iz + 1]

    c00 = c000 * (1.0 - tz) + c001 * tz
    c01 = c010 * (1.0 - tz) + c011 * tz
    c10 = c100 * (1.0 - tz) + c101 * tz
    c11 = c110 * (1.0 - tz) + c111 * tz
    c0 = c00 * (1.0 - ty) + c01 * ty
    c1 = c10 * (1.0 - ty) + c11 * ty
    return c0 * (1.0 - tx) + c1 * tx


@njit(cache=True)
def build_interp3_stencil_numba(
    x,
    y,
    z,
    r_grid: np.ndarray,
    q_grid: np.ndarray,
    k_grid: np.ndarray,
):
    """Precompute trilinear interpolation indices and barycentric weights."""
    r_min, r_max = r_grid[0], r_grid[-1]
    q_min, q_max = q_grid[0], q_grid[-1]
    k_min, k_max = k_grid[0], k_grid[-1]

    r_step = r_grid[1] - r_grid[0]
    q_step = q_grid[1] - q_grid[0]
    k_step = k_grid[1] - k_grid[0]

    x_clipped = clip_numba(x, r_min, r_max)
    y_clipped = clip_numba(y, q_min, q_max)
    z_clipped = clip_numba(z, k_min, k_max)

    ix = np.floor((x_clipped - r_min) / r_step).astype(np.int64)
    iy = np.floor((y_clipped - q_min) / q_step).astype(np.int64)
    iz = np.floor((z_clipped - k_min) / k_step).astype(np.int64)

    ix = np.minimum(np.maximum(ix, 0), len(r_grid) - 2)
    iy = np.minimum(np.maximum(iy, 0), len(q_grid) - 2)
    iz = np.minimum(np.maximum(iz, 0), len(k_grid) - 2)

    tx = (x_clipped - (r_min + ix * r_step)) / r_step
    ty = (y_clipped - (q_min + iy * q_step)) / q_step
    tz = (z_clipped - (k_min + iz * k_step)) / k_step
    return ix, iy, iz, tx, ty, tz


def build_interp3_stencil(
    x,
    y,
    z,
    r_grid: np.ndarray,
    q_grid: np.ndarray,
    k_grid: np.ndarray,
):
    return build_interp3_stencil_numba(x, y, z, r_grid, q_grid, k_grid)


@njit(cache=True)
def interp3_from_stencil_numba(values, ix, iy, iz, tx, ty, tz):
    out = np.empty(ix.shape, dtype=np.float64)
    out_flat = out.ravel()
    ix_flat = ix.ravel()
    iy_flat = iy.ravel()
    iz_flat = iz.ravel()
    tx_flat = tx.ravel()
    ty_flat = ty.ravel()
    tz_flat = tz.ravel()

    for n in range(out_flat.size):
        i = ix_flat[n]
        j = iy_flat[n]
        k = iz_flat[n]
        wx = tx_flat[n]
        wy = ty_flat[n]
        wz = tz_flat[n]

        c000 = values[i, j, k]
        c001 = values[i, j, k + 1]
        c010 = values[i, j + 1, k]
        c011 = values[i, j + 1, k + 1]
        c100 = values[i + 1, j, k]
        c101 = values[i + 1, j, k + 1]
        c110 = values[i + 1, j + 1, k]
        c111 = values[i + 1, j + 1, k + 1]

        c00 = c000 * (1.0 - wz) + c001 * wz
        c01 = c010 * (1.0 - wz) + c011 * wz
        c10 = c100 * (1.0 - wz) + c101 * wz
        c11 = c110 * (1.0 - wz) + c111 * wz
        c0 = c00 * (1.0 - wy) + c01 * wy
        c1 = c10 * (1.0 - wy) + c11 * wy
        out_flat[n] = c0 * (1.0 - wx) + c1 * wx

    return out


def interp3_from_stencil(values: np.ndarray, stencil) -> np.ndarray:
    return interp3_from_stencil_numba(values, *stencil)
