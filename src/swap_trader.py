#!/usr/bin/env python3
"""Compatibility facade for the swap trading solver."""
from __future__ import annotations

try:
    from .solver import Params, Solver
except ImportError:
    from solver import Params, Solver


solver = Solver()

P = solver.params
FIXED_RATE = solver.fixed_rate

NR = solver.nr
NQ = solver.nq
NK = solver.nk

R_GRID = solver.r_grid
Q_GRID = solver.q_grid
K_GRID = solver.k_grid
R_MESH = solver.r_mesh
Q_MESH = solver.q_mesh
K_MESH = solver.k_mesh

PHI = solver.phi
BASE_SD = solver.base_sd
Z_NODES = solver.z_nodes
Z_PROBS = solver.z_probs

R_MIN = solver.r_min
R_MAX = solver.r_max
Q_MIN = solver.q_min
Q_MAX = solver.q_max
K_MIN = solver.k_min
K_MAX = solver.k_max

R_STEP = solver.r_step
Q_STEP = solver.q_step
K_STEP = solver.k_step
DISC_GRID = solver.disc_grid


def interp3_uniform(values, x, y, z):
    return solver.interp3_uniform(values, x, y, z)


def build_interp3_stencil(x, y, z):
    return solver.build_interp3_stencil(x, y, z)


def interp3_from_stencil(values, stencil):
    return solver.interp3_from_stencil(values, stencil)


def get_next_position(q, u):
    return solver.get_next_position(q, u)


def get_flows_on_grid(t, u):
    return solver.get_flows_on_grid(t, u)


def get_flows(t, r, q, K, u):
    return solver.get_flows(t, r, q, K, u)


def get_delta_K_on_grid(t, u):
    return solver.get_delta_K_on_grid(t, u)


def get_delta_K_bar_on_grid(t, u):
    return solver.get_delta_K_bar_on_grid(t, u)


def get_delta_K(t, r, q, K, u):
    return solver.get_delta_K(t, r, q, K, u)


def get_delta_K_bar(t, r, q, K, u):
    return solver.get_delta_K_bar(t, r, q, K, u)


def horizon_to_steps(horizon):
    return solver.horizon_to_steps(horizon)


def diffuse(surf, qn, Kn):
    return solver.diffuse(surf, qn, Kn)


def diffuse_1d(y):
    return solver.diffuse_1d(y)


def get_swap_mtm():
    return solver.get_swap_mtm()


def build_action_cache():
    return solver.build_action_cache()


def diffuse_cached(surf, stencils):
    return solver.diffuse_cached(surf, stencils)


def get_optimal_policy():
    return solver.get_optimal_policy()


def get_optimal_policy_capital(horizon):
    return solver.get_optimal_policy_capital(horizon)


def get_optimal_policy_capital_short(horizon):
    return solver.get_optimal_policy_capital_short(horizon)


def get_capital(policy, horizon):
    return solver.get_capital(policy, horizon)


def get_value(policy):
    return solver.get_value(policy)


def get_scenario(rate, q0, K0, policy):
    return solver.get_scenario(rate, q0, K0, policy)


SWAP_MTM = solver.swap_mtm
ACTION_CACHE = solver.action_cache
