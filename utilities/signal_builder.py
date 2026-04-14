"""
signal_builder.py
=================
Signal reconstruction and unit conversions.

Exact Python translation of:
    sumSignalDroso.py  (sumSignal1_par)
    singlePolSignal.py (Signal_par)
by Ovidiu Radulescu / Maria Douaihy, University of Montpellier.

Verified to machine precision (max diff < 1e-14) against stored
DataPred in real BurstDECONV NPZ output files.

Time conversion verified against common_part_fitting__12_.py line 339-340:
    times_min = (index + 1) / FreqEchSimu / 60
"""

import numpy as np


def Signal_par(ypos, Intensity_for_1_Polym, TailleSeqMarq):
    """
    Fluorescence from one polymerase.
    Exact translation of singlePolSignal.py.
    """
    S    = np.ones(len(ypos)) * Intensity_for_1_Polym
    ind2 = np.where(ypos < TailleSeqMarq)[0]
    S[ind2] = (1 + ypos[ind2]) / TailleSeqMarq * Intensity_for_1_Polym
    return S


def build_signal(binary_vec, hp, frame_num):
    """
    Reconstruct MS2 fluorescence from binary initiation vector.
    Exact translation of sumSignalDroso.py :: sumSignal1_par().

    Parameters
    ----------
    binary_vec : np.ndarray of int (0/1), shape (num_poly,)
    hp         : dict from load_hyperparams()
    frame_num  : int

    Returns
    -------
    signal : np.ndarray of float, shape (frame_num,)
    """
    Polym_speed           = hp['Polym_speed']
    TaillePreMarq         = hp['TaillePreMarq']
    TailleSeqMarq         = hp['TailleSeqMarq']
    TaillePostMarq        = hp['TaillePostMarq']
    FreqEchImg            = hp['FreqEchImg']
    FreqEchSimu           = hp['FreqEchSimu']
    Intensity_for_1_Polym = hp['Intensity_for_1_Polym']
    Taille = TaillePreMarq + TailleSeqMarq + TaillePostMarq

    Trans_positions = np.where(binary_vec == 1)[0]
    if len(Trans_positions) == 0:
        return np.zeros(frame_num)

    # Vectorised — matches sumSignalDroso.py line by line
    Sum_signals_matrix = np.zeros((frame_num, len(Trans_positions)))

    ximage = (np.transpose(
                  np.tile(1 + np.arange(frame_num),
                          (len(Trans_positions), 1))
              )) / FreqEchImg * Polym_speed

    xpos = (Trans_positions + 1) / FreqEchSimu * Polym_speed - Taille

    t1   = np.tile(xpos + TaillePreMarq, (frame_num, 1))
    ypos = ximage - t1

    ind  = np.logical_and(ypos > 0, ypos < (TailleSeqMarq + TaillePostMarq))
    Sum_signals_matrix[ind] = (Sum_signals_matrix[ind]
                                + Signal_par(ypos[ind] - 1,
                                             Intensity_for_1_Polym,
                                             TailleSeqMarq))
    return Sum_signals_matrix.sum(axis=1)


def binary_to_times_min(binary_vec, FreqEchSimu):
    """
    Convert binary initiation vector to real initiation times in minutes.

    Matches common_part_fitting__12_.py lines 339-340:
        times     = (index+1)/FreqEchSimu - Taille/Polym_speed
        times_min = (times + Taille/Polym_speed) / 60
                  = (index+1) / FreqEchSimu / 60   [Taille terms cancel]

    Parameters
    ----------
    binary_vec  : np.ndarray of int (0/1)
    FreqEchSimu : float

    Returns
    -------
    times_min : np.ndarray of float
    """
    indices = np.where(binary_vec == 1)[0]
    return (indices + 1) / FreqEchSimu / 60.0
