# noise_rms.py
#
# Computes a RMS noise map per-pixel from a multi-extension FITS file after excluding dead channels.
# Each HDU contains a (2, 4096, 4096) cube; so we do CDS = scan1 - scan0.
# (scan0 is baseline)
# The RMS map is the standard deviation of CDS frames over all ~100 frames.
#
# Usage:
#   /disk/bifrost/uvexdet/miniconda3/bin/python3 noise_rms.py <input.fits>                      # uses GAINFITS or fallback_gain value, and output_dir
#   /disk/bifrost/uvexdet/miniconda3/bin/python3 noise_rms.py <input.fits> --gain 1.131         # takes gain value as input, uses output_dir
#   /disk/bifrost/uvexdet/miniconda3/bin/python3 noise_rms.py <input.fits> --outdir /some/path/ # uses fallback_gain value, takes outdir as input
#   example: /disk/bifrost/uvexdet/miniconda3/bin/python3 /disk/bifrost/nkerkese/noise_rms.py /disk/bifrost/nkerkese/260629/260630_012720_noise_hg.fits --outdir /disk/bifrost/nkerkese/260629/ --temp 166 --gain 1.09
#
# Outputs (all saved to output_dir/<basename>/):
#   <basename>_cds.fits            — CDS stack (scan1 - scan0) used to compute RMS
#   <basename>_rms.fits            — per-pixel RMS map
#   <basename>_rms.png              — display image of RMS map, with clean boxes overlaid
#   <basename>_hist.png             — histogram of RMS values (clean-box pixels only), log + linear
#   <basename>_noise_fraction.png   — RMS noise vs. fraction of pixels used (clean-box pixels only)
#   <basename>_stats.csv            — statistics table (clean-box pixels, before/after sigma clip)
#   <basename>_slide.png            — combined tile of hist + noise_fraction + rms map, for slides
#
# then download all pngs/csvs to Mac - see rsync command in pipeline notes


import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clip
import csv 
import os
import sys
import argparse
import matplotlib.patches as patches

# Parameters that can be changed

output_dir = "/disk/bifrost/nkerkese/"
temp_keyword = "TEMP"
sigma_clip_value = 3 # can be 3,5,...
sigma_max_iterations = 10 # sigma clipping iterations
skip_frames  = 5     # number of frames to skip from the beginning
max_frames = 10  # to reduce run time if needed

# incase frame orders change
scan0_idx = 0
scan1_idx = 1

# vmin/vmax adjsuting for RMS map
vmin_pct = 1.0
vmax_pct = 99.0

gain_keyword = 'GAINFITS'
fallback_gain =  1.131 # GAINFITS=  1.131 e⁻/ADU (used only if --gain is not specified, header doesn't have GAINFITS)

parser = argparse.ArgumentParser()
parser.add_argument('--temp', type=float, default=None, help='Temperature value (overrides header)')
parser.add_argument('--gain', type=float, default=None, help='Overrides header value if given (Gain in e-/ADU)')
parser.add_argument('input_fits', help='input FITS file path')
parser.add_argument('--outdir', default=None, help='output directory')
args = parser.parse_args()


def get_metdata(hdul):

    if args.temp is not None:
        temp = args.temp
        print(f'using --temp value: {temp}')
    elif temp_keyword in hdul[1].header:
        temp = hdul[1].header[temp_keyword]
        print(f'using temp from header: {temp}')
    else:
        temp = None
        print(f" WARNING: '{temp_keyword}' doesn't exist.")

    if args.gain is not None:
        gain = args.gain
        print(f'using --gain value: {gain}')
        GAINUSER_value = True
    elif gain_keyword in hdul[1].header:
        gain = hdul[1].header[gain_keyword]
        print(f'using gain from header "{gain_keyword}" to  pull the gain value: {gain}')
        GAINUSER_value = False
    else:
        gain = fallback_gain
        print(f" --gain not specified, no gain keyword in the header,\n Using fallback gain value '{fallback_gain}")
        GAINUSER_value = False

    return temp, gain, GAINUSER_value


def load_cds_stack(hdul):
    """
    opens the fits stack file and retuns a stack of cds frames.
    CDS = scan1 - scan0
    (scan0 is the baseline)

    Returns:
        cds_stack : float32 array, shape (N_exposres, 4096, 4096)
        temp: float or None
    """
    cds_frames = []
    frames_to_load = hdul[1 + skip_frames:]
    if max_frames is not None:
        frames_to_load = frames_to_load[:max_frames]

    n_hdus = len(frames_to_load)
    for i, hdu in enumerate(frames_to_load):
        cube = hdu.data.astype(np.float32)
        cds = cube[scan1_idx] - cube[scan0_idx]
        cds_frames.append(cds)
        if (i+1) % 20 == 0 or (i+1) == n_hdus:
            print(f" Loaded {i+1}/{n_hdus} exposures")
    cds_stack = np.stack(cds_frames, axis=0)   # now we have stack instead of list
    print(f" CDS stack shape {cds_stack.shape}")
    return cds_stack

def compute_rms_map(cds_stack, gain):
    """
    per pixel sandart deviation over ~100 frames
    """
    rms_map_ADU = np.std(cds_stack, axis=0, ddof=1).astype(np.float32)
    rms_map = rms_map_ADU * gain
    return rms_map

def find_clean_boxes(rms_map):
    """
    Finds dead columns and rows in the RMS map and returns
    a list of clean box regions (as numpy slices) between them.
    """
    nrows, ncols = rms_map.shape

    # Find dead columns and rows (all pixels == 0)
    dead_cols = np.where(np.all(rms_map == 0, axis=0))[0]
    dead_rows = np.where(np.all(rms_map == 0, axis=1))[0]

    # Find contiguous groups of dead columns
    col_groups = []
    if len(dead_cols) > 0:
        start = dead_cols[0]
        for i in range(len(dead_cols) - 1):
            if dead_cols[i] + 1 != dead_cols[i+1]:
                col_groups.append((start, dead_cols[i]))
                start = dead_cols[i+1]
        col_groups.append((start, dead_cols[-1]))

    # Find contiguous groups of dead rows
    row_groups = []
    if len(dead_rows) > 0:
        start = dead_rows[0]
        for i in range(len(dead_rows) - 1):
            if dead_rows[i] + 1 != dead_rows[i+1]:
                row_groups.append((start, dead_rows[i]))
                start = dead_rows[i+1]
        row_groups.append((start, dead_rows[-1]))

    # Build clean column boundaries
    col_boundaries = [0]
    for start_col, end_col in col_groups:
        col_boundaries.append(start_col)
        col_boundaries.append(end_col + 1)
    col_boundaries.append(ncols)

    # Build clean row boundaries
    row_boundaries = [0]
    for start_row, end_row in row_groups:
        row_boundaries.append(start_row)
        row_boundaries.append(end_row + 1)
    row_boundaries.append(nrows)

    # Build clean boxes as numpy slices
    clean_boxes = []
    for r in range(0, len(row_boundaries) - 1, 2):
        for c in range(0, len(col_boundaries) - 1, 2):
            row_slice = slice(row_boundaries[r], row_boundaries[r+1])
            col_slice = slice(col_boundaries[c], col_boundaries[c+1])
            clean_boxes.append((row_slice, col_slice))

    print(f"Found {len(col_groups)} dead column channel(s), {len(row_groups)} dead row channel(s)")
    print(f"Created {len(clean_boxes)} clean box(es)")
    for i, box in enumerate(clean_boxes):
        print(f"  Box {i+1}: rows {box[0].start}-{box[0].stop}, cols {box[1].start}-{box[1].stop}")

    return clean_boxes


def compute_stats(rms, label, sigma_val, max_iter, temp, GAINUSER_value, gain):
    """
    computes stats from the RMS, before and after sigma clipping
    """
    rows = []
    # before clipping
    rows.append({
        'label'      : label,
        'stage'      : 'no_clipping',
        'GAINUSER'   : GAINUSER_value,
        'Gain'       : float(gain),
        'mean'       : float(np.mean(rms)),
        'median'     : float(np.median(rms)),
        'std'        : float(np.std(rms, ddof=1)),
        'sigma_clip' : sigma_val,
        'n_pixels'   : len(rms),
        'n_clipped'  : 0,
        'temp'       : temp if temp is not None else float('nan'),
    })
    
    #after clipping
    clipped = sigma_clip(rms, sigma=sigma_val, maxiters=max_iter, masked=True)
    good    = rms[~clipped.mask]
    rows.append({
        'label'      : label,
        'stage'      : f'sigma{sigma_val}',
        'GAINUSER'   : GAINUSER_value,
        'Gain'       : float(gain),
        'mean'       : float(np.mean(good)),
        'median'     : float(np.median(good)),
        'std'        : float(np.std(good, ddof=1)),
        'sigma_clip' : sigma_val,
        'n_pixels'   : len(good),
        'n_clipped'  : int(np.sum(clipped.mask)), # since True = 1 and False = 0
        'temp'       : temp if temp is not None else float('nan'),
    })

    return rows

# print valeus
def print_stats(rows):
    for r in rows:
        print(f"\n  [{r['label']} | {r['stage']}]")
        print(f"    Mean   : {r['mean']:.4f}")
        print(f"    Median : {r['median']:.4f}")
        print(f"    Std    : {r['std']:.4f}")
        print(f"    N px   : {r['n_pixels']}  (clipped: {r['n_clipped']})")
        print(f"    Temp   : {r['temp']}")

def save_rms_fits(rms_map, fits_path, out_path):
    """
    Save RMS map as a FITS file, copying primary header from input.
    """
    with fits.open(fits_path) as hdul:
        primary_header = hdul[0].header.copy()
    primary_header['HISTORY'] = 'RMS map computed by noise_rms.py'
    primary_header['NOISEMAP'] = 'per-pixel std of CDS frames'
    hdu = fits.PrimaryHDU(data=rms_map, header=primary_header)
    hdu.writeto(out_path, overwrite=True)
    print(f"Saved RMS FITS: {out_path}")

def save_rms_image(rms_map, out_path, temp=None, clean_boxes=None):
    import matplotlib.patches as patches
    vmin = np.percentile(rms_map, vmin_pct)
    vmax = np.percentile(rms_map, vmax_pct)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(rms_map, origin='lower', cmap='inferno',
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    plt.colorbar(im, ax=ax, label='RMS (e⁻)')
    ax.set_title(f'Per-pixel RMS Map (CDS), Temp = {temp}')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')

    if clean_boxes is not None:
        for box in clean_boxes:
            x      = box[1].start
            y      = box[0].start
            width  = box[1].stop - box[1].start
            height = box[0].stop - box[0].start
            rect = patches.Rectangle((x, y), width, height,
                                      linewidth=1.5, edgecolor='red',
                                      facecolor='none', linestyle='--')
            ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved RMS image: {out_path}")


def save_histogram(rms_flat, stat_rows, out_path, label='Full Image', temp=None):
    """
    Save histogram of RMS values, with before/after sigma clip marked.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    raw_row    = next(r for r in stat_rows if r['stage'] == 'no_clipping')
    clip_row   = next(r for r in stat_rows if r['stage'] != 'no_clipping')
    clip_sigma = clip_row['sigma_clip']

    # for fraction of pixels
    rms_sorted = np.sort(rms_flat)
    N = len(rms_sorted)
    cumulative_fraction = np.arange(1, N+1)/N * 100

    xmin = rms_flat[rms_flat > 0].min()   # exclude zeros
    xmax = rms_flat.max()
    xmin_log = max(xmin, 1e-6)


    std_all   = np.std(rms_flat)
    mean_all  = np.mean(rms_flat)
    xmax_linear = mean_all + 3 * std_all
    bins_log = np.logspace(np.log10(xmin_log), np.log10(xmax), 100)
    bins = np.linspace(xmin, xmax_linear, 100)

    # log plot
    ax[0].hist(rms_flat,    bins=bins_log, alpha=0.7, label='All pixels')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].axvline(raw_row['mean'], color='orange', lw=1.5, linestyle='--', label=f"Mean (raw) = {raw_row['mean']:.2f}")
    ax[0].axvline(clip_row['mean'], color='pink', lw=1.5, linestyle='-', label=f"Mean ({clip_sigma}σ clip) = {clip_row['mean']:.2f}")
    ax[0].set_xlabel('RMS (e⁻)')
    ax[0].set_ylabel('Number of pixels')
    ax02 = ax[0].twinx()
    ax02.plot(rms_sorted, cumulative_fraction, color='red', label='Cumulative')
    ax02.set_ylim(0, 100)
    ax02.set_ylabel('Fraction of pixels (%)')
    ax[0].set_xlim(xmin, xmax)
    ax[0].set_title(f'RMS Distribution — {label}')
    lines0, labels0 = ax[0].get_legend_handles_labels()
    lines02, labels02 = ax02.get_legend_handles_labels()
    ax[0].legend(lines0 + lines02, labels0 + labels02, fontsize=9, loc='upper right')

    # linear plot
    ax[1].hist(rms_flat, bins=bins, alpha=0.7, label='All pixels')
    ax[1].set_title(f"Median = {raw_row['median']:2q.2f} RMS e⁻, Temp = {temp}")
    ax[1].axvline(raw_row['mean'],  color='orange', lw=1.5, linestyle='--', label=f"Mean (raw) = {raw_row['mean']:.2f}")
    ax[1].axvline(clip_row['mean'], color='pink',    lw=1.5, linestyle='-', label=f"Mean ({clip_sigma}σ clip) = {clip_row['mean']:.2f}")
    ax[1].set_xlabel('RMS (e⁻)')
    ax[1].set_ylabel('Number of pixels')
    ax[1].set_xlim(xmin, xmax_linear)
    ax12 = ax[1].twinx()
    ax12.plot(rms_sorted, cumulative_fraction, color='red', label='Cumulative')
    ax12.set_ylim(0, 100)
    ax12.set_ylabel('Fraction of pixels (%)')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved histogram: {out_path}")


def save_csv(all_rows, out_path):
    """
    Writes statistics rows to a CSV file.
    """
    fieldnames = ['label', 
                  'stage', 
                  'GAINUSER', 
                  'Gain',    
                  'mean', 
                  'median', 
                  'std',
                  'sigma_clip', 
                  'n_pixels', 
                  'n_clipped', 
                  'temp']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved CSV: {out_path}")

def save_cds_fits(cds_stack, fits_path, out_path):
    """
    Save CDS stack as a FITS file.
    Shape: (N_exposures, 4096, 4096)
    """
    with fits.open(fits_path) as hdul:
        primary_header = hdul[0].header.copy()
    primary_header['HISTORY'] = 'CDS stack computed by noise_rms.py'
    primary_header['CDSNOTE'] = 'CDS = scan1 - scan0 per exposure'
    hdu = fits.PrimaryHDU(data=cds_stack, header=primary_header)
    hdu.writeto(out_path, overwrite=True)
    print(f"Saved CDS FITS: {out_path}")


def save_noise_fraction_plot(rms_flat, out_path, temp=None):
    """
    Plot mean RMS noise vs fraction of pixels.
    """
    rms_sorted = np.sort(rms_flat)
    N = len(rms_sorted)

    # percentage cutoff parameters
    pct_low  = 0    # cut bottom X% (0 = no cut)
    pct_high = 0    # cut top X% (0 = no cut)


    n_low  = int(N * pct_low  / 100)
    n_high = int(N * (1 - pct_high / 100))
    rms_trimmed = rms_sorted[n_low:n_high]
    N_trimmed = len(rms_trimmed)

    percentages = np.linspace(1, 100, 200)
    mean_rms = []

    for pct in percentages:
        n_keep = int(N_trimmed * pct / 100)
        subset = rms_trimmed[:n_keep]
        if len(subset) == 0:
            mean_rms.append(np.nan)
        else:
            mean_rms.append(np.mean(subset))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(percentages, mean_rms, lw=1.5)
    ax.set_xlabel('Fraction of pixels used (%)')
    ax.set_ylabel('RMS Noise (e⁻)')
    ax.set_title(f'RMS Noise vs Fraction of Pixels Used ({pct_low}% cut bottom, {pct_high}% cut top), Temp = {temp}')    
    ax.set_xticks(np.arange(0, 101, 5))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved noise fraction plot: {out_path}")

# ------- for slides ---------------------------------------------------------------------------

def save_slide_summary(rms_map, rms_flat, stat_rows, clean_boxes, out_path,
                        temp=None, gain=None, device_label=None):
    """
    Combines hist (log+linear), noise fraction, and RMS map into one tiled PNG
    for slide presentation. Duplicates plotting logic from save_histogram,
    save_noise_fraction_plot, and save_rms_image (kept independent so the
    standalone PNGs are unaffected by changes here and this can be commented out)
    """

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.6])
    #gs = fig.add_gridspec(2, 3)

    ax_hist_log   = fig.add_subplot(gs[0, 0])
    ax_hist_lin   = fig.add_subplot(gs[0, 1])
    ax_noise_frac = fig.add_subplot(gs[1, 0:2])
    ax_rms_map    = fig.add_subplot(gs[:, 2])

    raw_row    = next(r for r in stat_rows if r['stage'] == 'no_clipping')
    clip_row   = next(r for r in stat_rows if r['stage'] != 'no_clipping')
    clip_sigma = clip_row['sigma_clip']

    rms_sorted = np.sort(rms_flat)
    N = len(rms_sorted)
    cumulative_fraction = np.arange(1, N+1)/N * 100

    xmin = rms_flat[rms_flat > 0].min()
    xmax = rms_flat.max()
    xmin_log = max(xmin, 1e-6)
    bins_log = np.logspace(np.log10(xmin_log), np.log10(xmax), 100)

    std_all  = np.std(rms_flat)
    mean_all = np.mean(rms_flat)
    xmax_linear = mean_all + 3 * std_all
    bins = np.linspace(xmin, xmax_linear, 100)

    # --- log histogram ---
    ax_hist_log.hist(rms_flat, bins=bins_log, alpha=0.7, label='All pixels')
    ax_hist_log.set_xscale('log')
    ax_hist_log.set_yscale('log')
    ax_hist_log.axvline(raw_row['mean'], color='orange', lw=1.5, linestyle='--',
                         label=f"Mean (raw) = {raw_row['mean']:.2f}")
    ax_hist_log.axvline(clip_row['mean'], color='pink', lw=1.5, linestyle='-',
                         label=f"Mean ({clip_sigma}σ clip) = {clip_row['mean']:.2f}")
    ax_hist_log.set_xlabel('RMS (e⁻)')
    ax_hist_log.set_ylabel('Number of pixels')
    ax_hist_log_twin = ax_hist_log.twinx()
    ax_hist_log_twin.plot(rms_sorted, cumulative_fraction, color='red', label='Cumulative')
    ax_hist_log_twin.set_ylim(0, 100)
    ax_hist_log_twin.set_ylabel('Fraction of pixels (%)')
    ax_hist_log.set_xlim(xmin, xmax)
    ax_hist_log.set_title('RMS Distribution — Full Image')
    lines0, labels0 = ax_hist_log.get_legend_handles_labels()
    lines02, labels02 = ax_hist_log_twin.get_legend_handles_labels()
    ax_hist_log.legend(lines0 + lines02, labels0 + labels02, fontsize=8, loc='upper right')

    # --- linear histogram ---
    ax_hist_lin.hist(rms_flat, bins=bins, alpha=0.7, label='All pixels')
    ax_hist_lin.set_title(f"Median = {raw_row['median']:.2f} RMS e⁻, Temp = {temp}")
    ax_hist_lin.axvline(raw_row['mean'], color='orange', lw=1.5, linestyle='--',
                         label=f"Mean (raw) = {raw_row['mean']:.2f}")
    ax_hist_lin.axvline(clip_row['mean'], color='pink', lw=1.5, linestyle='-',
                         label=f"Mean ({clip_sigma}σ clip) = {clip_row['mean']:.2f}")
    ax_hist_lin.set_xlabel('RMS (e⁻)')
    ax_hist_lin.set_ylabel('Number of pixels')
    ax_hist_lin.set_xlim(xmin, xmax_linear)
    ax_hist_lin_twin = ax_hist_lin.twinx()
    ax_hist_lin_twin.plot(rms_sorted, cumulative_fraction, color='red', label='Cumulative')
    ax_hist_lin_twin.set_ylim(0, 100)
    ax_hist_lin_twin.set_ylabel('Fraction of pixels (%)')

    # --- noise fraction plot ---
    pct_low  = 0
    pct_high = 0
    n_low  = int(N * pct_low  / 100)
    n_high = int(N * (1 - pct_high / 100))
    rms_trimmed = rms_sorted[n_low:n_high]
    N_trimmed = len(rms_trimmed)

    percentages = np.linspace(1, 100, 200)
    mean_rms = []
    for pct in percentages:
        n_keep = int(N_trimmed * pct / 100)
        subset = rms_trimmed[:n_keep]
        mean_rms.append(np.nan if len(subset) == 0 else np.mean(subset))

    ax_noise_frac.plot(percentages, mean_rms, lw=1.5)
    ax_noise_frac.set_xlabel('Fraction of pixels used (%)')
    ax_noise_frac.set_ylabel('RMS Noise (e⁻)')
    ax_noise_frac.set_title(f'RMS Noise vs Fraction of Pixels Used ({pct_low}% cut bottom, {pct_high}% cut top), Temp = {temp}')
    ax_noise_frac.set_xticks(np.arange(0, 101, 5))
    ax_noise_frac.grid(True, alpha=0.3)

    # --- RMS map ---
    vmin = np.percentile(rms_map, vmin_pct)
    vmax = np.percentile(rms_map, vmax_pct)
    im = ax_rms_map.imshow(rms_map, origin='lower', cmap='inferno',
                            vmin=vmin, vmax=vmax, interpolation='nearest')
    fig.colorbar(im, ax=ax_rms_map, label='RMS (e⁻)')
    ax_rms_map.set_title(f'Per-pixel RMS Map (CDS), Temp = {temp}')
    ax_rms_map.set_xlabel('X (pixels)')
    ax_rms_map.set_ylabel('Y (pixels)')

    if clean_boxes is not None:
        for box in clean_boxes:
            x      = box[1].start
            y      = box[0].start
            width  = box[1].stop - box[1].start
            height = box[0].stop - box[0].start
            rect = patches.Rectangle((x, y), width, height,
                                      linewidth=1.5, edgecolor='red',
                                      facecolor='none', linestyle='--')
            ax_rms_map.add_patch(rect)

    fig.suptitle(f"{device_label} - gain {gain} e/ADU, temp {temp} K", fontsize=18)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved slide summary: {out_path}")

#---------------------------------------------------------------------------------------------


def main():
    fits_path = args.input_fits
    out_dir   = args.outdir if args.outdir is not None else output_dir

    if not os.path.exists(fits_path):
        print(f"Error: File not found: {fits_path}")
        sys.exit(1)

    basename = os.path.splitext(os.path.basename(fits_path))[0]
    out_dir = os.path.join(out_dir, basename)
    os.makedirs(out_dir, exist_ok=True)

    # Output paths
    rms_fits_path   = os.path.join(out_dir, f"{basename}_rms.fits")
    rms_img_path    = os.path.join(out_dir, f"{basename}_rms.png")
    hist_path       = os.path.join(out_dir, f"{basename}_hist.png")
    csv_path        = os.path.join(out_dir, f"{basename}_stats.csv")
    cds_fits_path   = os.path.join(out_dir, f"{basename}_cds.fits")
    noise_frac_path = os.path.join(out_dir, f"{basename}_noise_fraction.png")
    slide_path = os.path.join(out_dir, f"{basename}_slide.png")
    #row_profile_path = os.path.join(out_dir, f"{basename}_row_profile.png")

    # Open fits file
    print(f'opening: {fits_path}')
    hdul = fits.open(fits_path, memmap=False)

    # Load and compute
    temp, gain, GAINUSER_value = get_metdata(hdul)
    cds_stack = load_cds_stack(hdul)
    rms_map   = compute_rms_map(cds_stack, gain)
    clean_boxes = find_clean_boxes(rms_map)
    rms_flat = np.concatenate([rms_map[box].flatten() for box in clean_boxes])
    hdul.close()

    # Save CDS before deleting
    save_cds_fits(cds_stack, fits_path, cds_fits_path)
    del cds_stack

    # Save outputs
    print("\nSaving outputs...")
    save_rms_fits(rms_map, fits_path, rms_fits_path)
    save_rms_image(rms_map, rms_img_path, temp=temp, clean_boxes=clean_boxes)


    # Statistics
    print("\nComputing statistics...")
    stat_rows = compute_stats(rms_flat, label='full_image', sigma_val=sigma_clip_value, max_iter=sigma_max_iterations, temp=temp,
                              GAINUSER_value=GAINUSER_value, gain=gain)
    print_stats(stat_rows)

    save_slide_summary(rms_map, rms_flat, stat_rows, clean_boxes, slide_path, temp=temp, gain=gain, device_label=basename)


    save_noise_fraction_plot(rms_flat, noise_frac_path, temp=temp)
    save_histogram(rms_flat, stat_rows, hist_path, label='Full Image', temp=temp)
    save_csv(stat_rows, csv_path)

    print("\nDone.")


if __name__ == '__main__':
    main()



