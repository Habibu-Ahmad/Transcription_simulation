"""
gillespie.py
============
Gillespie stochastic simulation for 2-state and 3-state M2 models.

Exact Python translation of:
    artificial_data_2States.m  (Next_2States.m + propensity_2States.m)
    artificial_data_3states_M2.m (NextM2.m + propensityM2.m)
by Ovidiu Radulescu, University of Montpellier, 2019.

The Markov chain starts in the ELONG state at t=0.
Every return to ELONG is recorded as a polymerase initiation.
No postmitotic lag — identical to their original simulation.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 2-STATE MODEL
# States:  0=OFF, 1=ON, 2=ELONG
# Reactions:
#   0: OFF -> ON    (k1p)
#   1: ON  -> OFF   (k1m)
#   2: ON  -> ELONG (kini)
# On reaching ELONG: record position, reset immediately to ON
# ---------------------------------------------------------------------------

_SM_2S = np.array([
    [-1,  1,  0],
    [ 1, -1, -1],
    [ 0,  0,  1],
], dtype=np.int64)


def _prop_2S(X, k1p, k1m, kini):
    R = np.array([k1p * X[0], k1m * X[1], kini * X[1]])
    return R, R.sum()


def simulate_2S(k1p, k1m, kini, num_poly, tmaxsh, tinter, rng):
    """
    2-state Gillespie for one cell. Exact translation of their MATLAB.

    Returns
    -------
    pos : np.ndarray of int8, shape (num_poly,), binary vector
    """
    pos = np.zeros(num_poly, dtype=np.int8)
    X   = np.array([0, 0, 1], dtype=np.int64)  # start in ELONG (state 3)
    t   = 0.0

    while t < tmaxsh:
        if X[2] == 1:
            # Record initiation position
            ipos = int(round(t / tinter))
            if 0 <= ipos < num_poly:
                pos[ipos] = 1
            # Reset immediately to ON (state 2)
            X = np.array([0, 1, 0], dtype=np.int64)
            continue

        R, Rtot = _prop_2S(X, k1p, k1m, kini)
        if Rtot == 0:
            break
        # Draw next reaction
        cumR = np.cumsum(R) / Rtot
        i    = int(np.searchsorted(cumR, rng.random()))
        X    = X + _SM_2S[:, i]
        # Draw waiting time
        if Rtot < 1000:
            t -= (1.0 / Rtot) * np.log(rng.random())

    return pos


# ---------------------------------------------------------------------------
# 3-STATE M2 MODEL
# States:  0=OFF1, 1=PAUSE, 2=ON, 3=ELONG
# Reactions:
#   0: OFF1  -> ON    (k1p)
#   1: ON    -> OFF1  (k1m)
#   2: PAUSE -> ON    (k2p)
#   3: ON    -> PAUSE (k2m)
#   4: ON    -> ELONG (kini)
# On reaching ELONG: record position, reset immediately to ON
# ---------------------------------------------------------------------------

_SM_M2 = np.array([
    [-1,  1,  0,  0,  0],
    [ 0,  0, -1,  1,  0],
    [ 1, -1,  1, -1, -1],
    [ 0,  0,  0,  0,  1],
], dtype=np.int64)


def _prop_M2(X, k1p, k1m, k2p, k2m, kini):
    R = np.array([
        k1p  * X[0],
        k1m  * X[2],
        k2p  * X[1],
        k2m  * X[2],
        kini * X[2],
    ])
    return R, R.sum()


def simulate_M2(k1p, k1m, k2p, k2m, kini, num_poly, tmaxsh, tinter, rng):
    """
    3-state M2 Gillespie for one cell. Exact translation of their MATLAB.

    Returns
    -------
    pos : np.ndarray of int8, shape (num_poly,), binary vector
    """
    pos = np.zeros(num_poly, dtype=np.int8)
    X   = np.array([0, 0, 0, 1], dtype=np.int64)  # start in ELONG (state 4)
    t   = 0.0

    while t < tmaxsh:
        if X[3] == 1:
            ipos = int(round(t / tinter))
            if 0 <= ipos < num_poly:
                pos[ipos] = 1
            # Reset immediately to ON (state 3)
            X = np.array([0, 0, 1, 0], dtype=np.int64)
            continue

        R, Rtot = _prop_M2(X, k1p, k1m, k2p, k2m, kini)
        if Rtot == 0:
            break
        cumR = np.cumsum(R) / Rtot
        i    = int(np.searchsorted(cumR, rng.random()))
        X    = X + _SM_M2[:, i]
        if Rtot < 1000:
            t -= (1.0 / Rtot) * np.log(rng.random())

    return pos


def run_gillespie(burst_row, sim, rng):
    """
    Dispatch to correct model based on n_states in burst_row.

    Parameters
    ----------
    burst_row : pandas Series, one row from burst_best_fits.xlsx
    sim       : dict from derive_sim_constants()
    rng       : numpy.random.Generator

    Returns
    -------
    pos : np.ndarray of int8, shape (num_poly,)
    """
    import pandas as pd

    n_states = int(burst_row['n_states'])
    k1p  = float(burst_row['k1p'])
    k1m  = float(burst_row['k1m'])
    kini = float(burst_row['kin'])

    if n_states == 2:
        return simulate_2S(k1p, k1m, kini,
                           sim['num_poly'], sim['tmaxsh'], sim['tinter'], rng)
    else:
        k2p = float(burst_row['k2p'])
        k2m = float(burst_row['k2m'])
        return simulate_M2(k1p, k1m, k2p, k2m, kini,
                           sim['num_poly'], sim['tmaxsh'], sim['tinter'], rng)
