"""
hyperparams.py
==============
Load HyperParameters.npz and derive simulation constants.
"""

import os
import numpy as np


def load_hyperparams(npz_path):
    """Load HyperParameters.npz as a plain dict of floats."""
    data = np.load(npz_path)
    return {k: float(data[k]) for k in data.files}


def find_hyperparams_path(full_path):
    """
    Find HyperParameters.npz starting from full_path itself,
    then walking up the directory tree.
    """
    path = str(full_path)

    # Check inside the full_path folder itself first
    candidate = os.path.join(path, 'HyperParameters.npz')
    if os.path.exists(candidate):
        return candidate

    # Walk upward up to 7 levels
    for _ in range(7):
        path = os.path.dirname(path)
        candidate = os.path.join(path, 'HyperParameters.npz')
        if os.path.exists(candidate):
            return candidate

    return None


def derive_sim_constants(hp, movie_min):
    """
    Compute all derived simulation constants from hyperparams dict.

    Parameters
    ----------
    hp        : dict from load_hyperparams()
    movie_min : float, movie duration in minutes

    Returns
    -------
    dict with keys: tinter, DureeSignal, DureeSimu, DureeAnalysee,
                    num_poly, frame_num, tmaxsh
    """
    Polym_speed        = hp['Polym_speed']
    TaillePreMarq      = hp['TaillePreMarq']
    TailleSeqMarq      = hp['TailleSeqMarq']
    TaillePostMarq     = hp['TaillePostMarq']
    EspaceInterPolyMin = hp['EspaceInterPolyMin']
    FrameLen           = hp['FrameLen']

    tinter        = EspaceInterPolyMin / Polym_speed
    DureeSignal   = (TaillePreMarq + TailleSeqMarq + TaillePostMarq) / Polym_speed
    movie_s       = movie_min * 60.0
    lframes       = round(movie_s / FrameLen)
    DureeSimu     = lframes * FrameLen
    DureeAnalysee = DureeSignal + DureeSimu
    num_poly      = round(DureeAnalysee / tinter)
    frame_num     = lframes

    return dict(
        tinter        = tinter,
        DureeSignal   = DureeSignal,
        DureeSimu     = DureeSimu,
        DureeAnalysee = DureeAnalysee,
        num_poly      = int(num_poly),
        frame_num     = int(frame_num),
        tmaxsh        = DureeAnalysee,
    )
