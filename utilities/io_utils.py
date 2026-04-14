"""
io_utils.py
===========
Save output files for one dataset:
    RawData.xlsx           — MS2 fluorescence signal  (frames × nuclei)
    PosPred.xlsx           — binary initiation vectors (slots × nuclei)
    InitiationTimes.xlsx   — initiation times in minutes (events × nuclei)
    KineticParameters.xlsx — full parameter rows
    signals_panel_N.pdf    — 6×6 trace panels (style of common_part_fitting__12_.py)
    signal_heatmap.pdf     — signal intensity heatmap (jet colormap)
    pospred_heatmap.pdf    — PosPred density heatmap (YlOrBr colormap)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# EXCEL OUTPUTS
# ---------------------------------------------------------------------------

def save_rawdata(folder, signal_array, frame_len, filename='RawData.xlsx'):
    """Rows=frames, Columns=nuclei, plus time_s and time_min columns."""
    frame_num = signal_array.shape[0]
    n_cells   = signal_array.shape[1]
    cols      = [f'nucleus_{i:03d}' for i in range(n_cells)]
    t_s       = np.arange(frame_num) * frame_len
    t_min     = t_s / 60.0
    df        = pd.DataFrame(signal_array, columns=cols)
    df.insert(0, 'time_min', np.round(t_min, 6))
    df.insert(0, 'time_s',   np.round(t_s,   4))
    df.to_excel(os.path.join(folder, filename), index=False)


def save_pospred(folder, binary_list, FreqEchSimu):
    """Rows=slots, Columns=nuclei, plus time_min column."""
    num_poly = len(binary_list[0])
    n_cells  = len(binary_list)
    cols     = [f'nucleus_{i:03d}' for i in range(n_cells)]
    mat      = np.column_stack(binary_list).astype(np.int8)
    t_min    = np.round((np.arange(num_poly) + 1) / FreqEchSimu / 60.0, 6)
    df       = pd.DataFrame(mat, columns=cols)
    df.insert(0, 'time_min', t_min)
    df.to_excel(os.path.join(folder, 'PosPred.xlsx'), index=False)


def save_initiation_times(folder, times_list):
    """Rows=events (NaN-padded), Columns=nuclei."""
    n_cells = len(times_list)
    max_len = max(len(t) for t in times_list) if times_list else 0
    cols    = [f'nucleus_{i:03d}' for i in range(n_cells)]
    mat     = np.full((max_len, n_cells), np.nan)
    for i, t in enumerate(times_list):
        mat[:len(t), i] = t
    pd.DataFrame(mat, columns=cols).to_excel(
        os.path.join(folder, 'InitiationTimes.xlsx'), index=False
    )


def save_kinetic_params(folder, burst_row, postmit_row=None):
    """
    Save the full parameter rows from burst_best_fits.xlsx.
    If postmit_row is provided, append it below with a blank separator.
    """
    kin_df = burst_row.to_frame().T.reset_index(drop=True)
    kin_df.insert(0, 'block', 'kinetics')

    if postmit_row is not None:
        lag_df  = postmit_row.to_frame().T.reset_index(drop=True)
        lag_df.insert(0, 'block', 'postmitotic_lag')
        all_cols = list(dict.fromkeys(
            list(kin_df.columns) + list(lag_df.columns)
        ))
        kin_df = kin_df.reindex(columns=all_cols)
        lag_df = lag_df.reindex(columns=all_cols)
        sep    = pd.DataFrame([['' for _ in all_cols]], columns=all_cols)
        combined = pd.concat([kin_df, sep, lag_df], ignore_index=True)
    else:
        combined = kin_df

    combined.to_excel(
        os.path.join(folder, 'KineticParameters.xlsx'), index=False
    )


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

def plot_signals_panel(folder, signal_array, FreqEchImg, dataset_name):
    """
    6×6 trace panels matching common_part_fitting__12_.py style.
    - x-axis: time in minutes (0–40), from np.arange(frame_num)/FreqEchImg/60
    - black signal, linewidth=0.1
    - sns.despine(), font size 5
    - 36 nuclei per PDF page: signals_panel_1.pdf, signals_panel_2.pdf, ...
    """
    try:
        import seaborn as sns
        has_sns = True
    except ImportError:
        has_sns = False

    frame_num = signal_array.shape[0]
    n_cells   = signal_array.shape[1]
    t_min     = np.arange(frame_num) / FreqEchImg / 60.0

    n_per_fig = 36
    n_figs    = int(np.ceil(n_cells / n_per_fig))

    for fig_idx in range(n_figs):
        start    = fig_idx * n_per_fig
        end      = min(start + n_per_fig, n_cells)

        fig = plt.figure(figsize=(18, 15))
        plt.subplots_adjust(hspace=1.0, wspace=1.0)
        fig.suptitle(f'{dataset_name}  —  nuclei {start}–{end-1}',
                     fontsize=8, y=1.01)

        for local_idx, cell_idx in enumerate(range(start, end)):
            plt.subplot(6, 6, local_idx + 1)
            plt.plot(t_min, signal_array[:, cell_idx],
                     color='k', linewidth=0.1)
            plt.xlim(0, 40)
            plt.xticks(fontsize=5)
            plt.yticks(fontsize=5)
            if has_sns:
                sns.despine()

        figfile = os.path.join(folder, f'signals_panel_{fig_idx+1}.pdf')
        fig.savefig(figfile, bbox_inches='tight', dpi=150)
        plt.close(fig)


def plot_signal_heatmap(folder, signal_array, FreqEchImg, dataset_name):
    """
    Signal intensity heatmap (nuclei × time), jet colormap, square figure.
    Matches figure 50 in common_part_fitting__12_.py.
    """
    sz  = signal_array.shape
    X   = np.arange(0, sz[0]) / FreqEchImg / 60.0
    Y   = np.arange(1, sz[1] + 1)[::-1]
    extent = [X[0], X[-1], float(Y[-1]), float(Y[0])]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(signal_array.T, cmap='jet',
                   extent=extent, aspect='auto', origin='upper')
    ax.set_xlabel('Time [min]', fontsize=12)
    ax.set_ylabel('Transcription site', fontsize=12)
    ax.set_title(f'{dataset_name}  —  signal heatmap', fontsize=10)
    cb = fig.colorbar(im, ax=ax)
    cb.ax.tick_params(labelsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(folder, 'signal_heatmap.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_pospred_heatmap(folder, binary_list, hp, dataset_name):
    """
    PosPred density heatmap using seaborn YlOrBr colormap, square figure.
    Matches the PosPred figure in common_part_fitting__12_.py exactly:
    - 40-second time windows
    - X-axis shifted by DureeSignal to align to movie start
    - sns.heatmap with xticklabels=round(len(X)/6), yticklabels=round(n/4)
    """
    try:
        import seaborn as sns
        has_sns = True
    except ImportError:
        has_sns = False

    Polym_speed        = hp['Polym_speed']
    TaillePreMarq      = hp['TaillePreMarq']
    TailleSeqMarq      = hp['TailleSeqMarq']
    TaillePostMarq     = hp['TaillePostMarq']
    EspaceInterPolyMin = hp['EspaceInterPolyMin']
    FrameLen           = hp['FrameLen']

    PosPred  = np.column_stack(binary_list)   # (num_poly, n_cells)
    n_cells  = PosPred.shape[1]
    num_poly = PosPred.shape[0]

    # 40-second window in spacing slots
    time_window      = 40.0
    nbr_frame_in_win = max(1, int(time_window / FrameLen))
    n_blocks         = num_poly // nbr_frame_in_win + 1

    density = np.zeros((n_cells, n_blocks))
    for ii in range(n_cells):
        pp      = PosPred[:, ii]
        total   = n_blocks * nbr_frame_in_win
        pp_pad  = np.resize(pp, total)
        pp_pad[len(pp):] = 0
        pp_r    = pp_pad.reshape(n_blocks, nbr_frame_in_win)
        density[ii, :] = pp_r.sum(axis=1)

    # Time axis: shifted so t=0 is movie start (subtract DureeSignal)
    X = (np.arange(0, n_blocks * nbr_frame_in_win)
         * EspaceInterPolyMin / Polym_speed / 60.0
         - (TaillePreMarq + TailleSeqMarq + TaillePostMarq) / Polym_speed / 60.0)
    X = X[::nbr_frame_in_win]

    df = pd.DataFrame(density, columns=np.round(X, 1))

    fig, ax = plt.subplots(figsize=(10, 10))
    if has_sns:
        sns.heatmap(df, cmap='YlOrBr', ax=ax,
                    xticklabels=max(1, round(len(X) / 6)),
                    yticklabels=max(1, round(n_cells / 4)))
    else:
        im = ax.imshow(density, cmap='YlOrBr', aspect='auto', origin='upper')
        fig.colorbar(im, ax=ax)

    ax.set_xlabel('Time [min]', fontsize=12)
    ax.set_ylabel('Transcription site', fontsize=12)
    ax.set_title(f'{dataset_name}  —  PosPred density heatmap  '
                 f'({n_cells} nuclei)', fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(folder, 'pospred_heatmap.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
