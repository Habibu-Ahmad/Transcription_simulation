"""
simulate.py
===========
Main orchestrator. Simulates 77 datasets from real biological parameters.

Usage in Colab
--------------
Upload all 4 files to /content/:
    hyperparams.py  gillespie.py  signal_builder.py  io_utils.py  simulate.py

Then run:

    from simulate import run_pipeline

    run_pipeline(
        burst_fits_path = "/content/drive/MyDrive/burst_best_fits.xlsx",
        output_dir      = "/content/drive/MyDrive/Datasets",
        n_cells         = 500,    # None = use real Nuclei count per dataset
        seed            = 42,
        movie_min       = None,   # None = derive from Frames * FrameLen
    )

Output structure
----------------
    Datasets/
    ├── metadata.xlsx          ← index of all 77 datasets
    ├── Data001/
    │   ├── RawData.xlsx
    │   ├── PosPred.xlsx
    │   ├── InitiationTimes.xlsx
    │   ├── HyperParameters.npz
    │   ├── KineticParameters.xlsx
    │   ├── signals_panel_1.pdf  (+ _2.pdf etc. if > 36 nuclei)
    │   ├── signal_heatmap.pdf
    │   └── pospred_heatmap.pdf
    ├── Data002/
    ...
    └── Data077/
"""

import os
import shutil
import numpy as np
import pandas as pd

from hyperparams    import load_hyperparams, find_hyperparams_path, derive_sim_constants
from gillespie      import run_gillespie
from signal_builder import build_signal, binary_to_times_min
from io_utils       import (save_rawdata, save_pospred, save_initiation_times,
                             save_kinetic_params, plot_signals_panel,
                             plot_signal_heatmap, plot_pospred_heatmap)


# ---------------------------------------------------------------------------
# SINGLE DATASET
# ---------------------------------------------------------------------------

def simulate_one_dataset(burst_row, folder, n_cells, rng,
                         movie_min_override=None):
    """
    Simulate one dataset and write all output files.

    Parameters
    ----------
    burst_row          : pandas Series, one row of burst_best_fits.xlsx
    folder             : str, output folder (will be created)
    n_cells            : int
    rng                : numpy.random.Generator
    movie_min_override : float or None

    Returns
    -------
    bool : True if successful, False if HyperParameters.npz not found
    """
    os.makedirs(folder, exist_ok=True)

    # --- hyperparameters ---
    hp_path = find_hyperparams_path(str(burst_row['full_path']))
    if hp_path is None or not os.path.exists(hp_path):
        print(f'    [SKIP] HyperParameters.npz not found')
        print(f'           Searched from: {burst_row["full_path"]}')
        return False

    hp = load_hyperparams(hp_path)
    shutil.copy2(hp_path, os.path.join(folder, 'HyperParameters.npz'))

    # --- movie duration ---
    if movie_min_override is not None:
        movie_min = float(movie_min_override)
    else:
        frames    = int(burst_row['Frames'])
        movie_min = frames * hp['FrameLen'] / 60.0

    sim         = derive_sim_constants(hp, movie_min)
    FreqEchSimu = hp['FreqEchSimu']
    FreqEchImg  = hp['FreqEchImg']
    frame_len   = hp['FrameLen']

    # --- simulate ---
    binary_list = []
    times_list  = []
    signals     = []

    for i in range(n_cells):
        bvec = run_gillespie(burst_row, sim, rng)
        sig  = build_signal(bvec, hp, sim['frame_num'])
        binary_list.append(bvec)
        times_list.append(binary_to_times_min(bvec, FreqEchSimu))
        signals.append(sig)

        if (i + 1) % 100 == 0 or (i + 1) == n_cells:
            print(f'    cell {i+1}/{n_cells}', end='\r')

    print()

    signal_array = np.column_stack(signals)

    # --- dataset label for plot titles ---
    ref_short = str(burst_row['reference']).split(' ')[0]
    dname     = (f"{ref_short} | {burst_row['gene']} | "
                 f"{burst_row['nc_cycle']} | {burst_row['region']} | "
                 f"{burst_row['phenotype']}")

    # --- save Excel ---
    save_rawdata(folder, signal_array, frame_len)
    save_pospred(folder, binary_list, FreqEchSimu)
    save_initiation_times(folder, times_list)
    save_kinetic_params(folder, burst_row)

    # --- save plots ---
    plot_signals_panel(folder, signal_array, FreqEchImg, dname)
    plot_signal_heatmap(folder, signal_array, FreqEchImg, dname)
    plot_pospred_heatmap(folder, binary_list, hp, dname)

    n_init_mean = np.mean([len(t) for t in times_list])
    print(f'    OK | {n_cells} cells | {sim["frame_num"]} frames | '
          f'{int(burst_row["n_states"])}S | '
          f'mean inits: {n_init_mean:.0f}')
    return True


# ---------------------------------------------------------------------------
# METADATA INDEX
# ---------------------------------------------------------------------------

def save_metadata(output_dir, burst_df, folder_names):
    """Save metadata.xlsx at the root of the output directory."""
    meta = burst_df.copy()
    meta.insert(0, 'folder', folder_names)
    meta.to_excel(os.path.join(output_dir, 'metadata.xlsx'), index=False)
    print(f'metadata.xlsx saved  ({len(meta)} datasets)')


# ---------------------------------------------------------------------------
# PIPELINE ENTRY POINT
# ---------------------------------------------------------------------------

def run_pipeline(burst_fits_path,
                 output_dir,
                 n_cells     = 100,
                 seed        = 42,
                 movie_min   = None):
    """
    Run the full simulation pipeline.

    Parameters
    ----------
    burst_fits_path : str
        Path to burst_best_fits.xlsx
    output_dir : str
        Root output directory (e.g. /content/drive/MyDrive/Datasets)
    n_cells : int or None
        Cells per dataset. None = use real Nuclei count from Excel file.
    seed : int
        Random seed. -1 for fully random.
    movie_min : float or None
        Movie duration in minutes.
        None = derive from Frames * FrameLen per dataset.
    """
    rng = np.random.default_rng(None if seed == -1 else seed)
    os.makedirs(output_dir, exist_ok=True)

    burst_df = pd.read_excel(burst_fits_path)

    print('=' * 65)
    print('BurstDECONV  —  Synthetic Dataset Generator')
    print('=' * 65)
    print(f'  Datasets      : {len(burst_df)}')
    print(f'  Cells/dataset : {n_cells or "from Nuclei column"}')
    print(f'  Output        : {output_dir}')
    print(f'  Seed          : {seed}')
    print()

    folder_names = []
    n_ok         = 0

    for idx, (_, burst_row) in enumerate(burst_df.iterrows()):

        folder_name = f'Data{idx+1:03d}'
        folder_path = os.path.join(output_dir, folder_name)
        folder_names.append(folder_name)

        cells = n_cells if n_cells is not None else max(1, int(burst_row['Nuclei']))

        ref_short = str(burst_row['reference']).split(' ')[0]
        print(f'[{folder_name}]  {ref_short} | {burst_row["gene"]} | '
              f'{burst_row["nc_cycle"]} | {burst_row["region"]} | '
              f'{burst_row["phenotype"]}')

        ok = simulate_one_dataset(
            burst_row, folder_path, cells, rng,
            movie_min_override=movie_min
        )
        if ok:
            n_ok += 1

    save_metadata(output_dir, burst_df, folder_names)

    print()
    print('=' * 65)
    print(f'Completed: {n_ok}/{len(burst_df)} datasets generated.')
    print(f'Output: {output_dir}')
    print('=' * 65)
