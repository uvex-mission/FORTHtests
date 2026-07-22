#!/usr/bin/env python3
"""
PTC (Photon Transfer Curve) Analysis Script

Reads one or more multi-extension FITS files matched by a shell glob / regexp
pattern.  Each extension must have an EXPTIME keyword (in seconds).
Extensions are grouped by EXPTIME **within each file**; each group must
contain exactly 2 frames.  Pairs never span file boundaries.

The difference method is used to separate shot noise from fixed-pattern noise:

    mean     = 0.5 * (mean(A) + mean(B))   within ROI
    variance = 0.5 * var(A − B)             within ROI

Mean–variance data from ALL files are combined for the PTC fit (gain, read
noise, full-well capacity).  The linearity plot colours each file differently.

Usage:
    python ptc_analysis.py "raw_*.fits" --roi X0 X1 Y0 Y1 [options]

"""

import argparse
import glob
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.io import fits
from scipy import stats

TIMEKEY = 'EXPTIME'  # FITS Key denoting exposure time
ADUKEY  = 'GAIN_F2E' # Multiply to convert ADUf to ADUe (electronics)


# 2.363 V 2.057 V  HG

# ---------------------------------------------------------------------------
# FITS loading — per file
# ---------------------------------------------------------------------------

def load_extensions(
    fits_path: Path,
    roi: Tuple[int, int, int, int],
    skip: int = 0,
    adu_convert: float = 1,
) -> List[Tuple[float, np.ndarray]]:
    """
    Open *fits_path* and return a list of (exptime_s, image_array) for every
    image-bearing extension after the first *skip* extensions (0-indexed).

    Each returned image is already cropped to *roi* = (x0, y0, x1, y1)
    (inclusive pixel coordinates).

    Extensions without a 2-D data array or without TIMEKEY are skipped with
    a warning.
    """
    results = []

    with fits.open(fits_path, memmap=False) as hdul:
        total = len(hdul)
        print(f"Opened {fits_path.name}  —  {total} HDU(s) total.")

        if skip >= total:
            sys.exit(
                f"Error: --skip {skip} is >= total number of HDUs ({total}) "
                f"in {fits_path.name}."
            )

        if skip:
            skipped_names = ", ".join(
                f"[{i}] {hdul[i].name}" for i in range(skip)
            )
            print(f"  Skipping first {skip} extension(s): {skipped_names}.")
        else:
            print(f"  Skipping first 0 extension(s).")

        for idx, hdu in enumerate(hdul):
            if idx < skip:
                continue

            # Must have 2-D image data
            if hdu.data is None or hdu.data.ndim < 2 or hdu.data.ndim > 3:
                print(f"  HDU[{idx}] '{hdu.name}': no 2-D data, skipping.")
                continue                

            # Must have TIMEKEY
            if TIMEKEY not in hdu.header:
                print(f"  HDU[{idx}] '{hdu.name}': no {TIMEKEY} keyword, skipping.")
                continue

            img = hdu.data.astype(np.float32) * adu_convert # Convert ADUf (FITS) to ADUe (electronics)

            # Apply ROI crop here so all downstream arrays are already trimmed
            # Do CDS subtraction if needed; crop first for efficiency
            if img.ndim == 3:
                roi3D = (slice(None),) + roi
                img = img[roi3D]
                img = img[1] - img[0]
            else:
                img = img[roi]

            exptime = float(hdu.header[TIMEKEY])
            results.append((exptime, img))

    if not results:
        print(f"Warning: no usable image extensions found in {fits_path.name} "
              f"after applying --skip.")

    return results


# ---------------------------------------------------------------------------
# Grouping into pairs (within a single file)
# ---------------------------------------------------------------------------

def group_into_pairs(
    extensions: List[Tuple[float, np.ndarray]],
    filename: str = "",
) -> Dict[float, List[np.ndarray]]:
    """
    Group images by TIMEKEY within a single file.  Returns only groups with
    >= 2 images; warns and skips groups with only 1.  If a group has > 2,
    uses the last two and warns.
    """
    groups: Dict[float, List[np.ndarray]] = defaultdict(list)
    for exptime, data in extensions:
        groups[exptime].append(data)

    pairs = {}
    tag = f" in {filename}" if filename else ""
    for exptime in sorted(groups):
        imgs = groups[exptime]
        if len(imgs) < 2:
            print(f"  Warning: {TIMEKEY}={exptime} s{tag} has only {len(imgs)} frame(s), skipping.")
        elif len(imgs) > 2:
            print(
                f"  Warning: {TIMEKEY}={exptime} s{tag} has {len(imgs)} frames "
                f"(expected 2); using last two."
            )
            pairs[exptime] = imgs[-2:]
        else:
            pairs[exptime] = imgs

    return pairs


# ---------------------------------------------------------------------------
# ROI and box partitioning
# ---------------------------------------------------------------------------

def parse_roi(args_roi: List[int]) -> Tuple[int, int, int, int]:
    x0, x1, y0, y1 = args_roi
    if x0 >= x1 or y0 >= y1:
        sys.exit("Error: ROI must satisfy X0 < X1 and Y0 < Y1.")
    return x0, y0, x1, y1


def trim_roi_to_boxsize(
    roi: Tuple[int, int, int, int],
    boxsize: int,
) -> Tuple[int, int, int, int]:
    """
    Shrink *roi* so its width and height are exact multiples of *boxsize*,
    keeping the top-left corner fixed.  Warns if trimming occurs.
    Returns the (possibly unchanged) roi.
    """
    x0, y0, x1, y1 = roi
    width  = x1 - x0 + 1
    height = y1 - y0 + 1

    new_width  = (width  // boxsize) * boxsize
    new_height = (height // boxsize) * boxsize

    if new_width == 0 or new_height == 0:
        sys.exit(
            f"Error: boxsize={boxsize} is larger than the ROI "
            f"({width}×{height} px).  Choose a smaller boxsize."
        )

    new_x1 = x0 + new_width  - 1
    new_y1 = y0 + new_height - 1

    if new_x1 != x1 or new_y1 != y1:
        trimmed_w = width  - new_width
        trimmed_h = height - new_height
        print(
            f"Note: ROI trimmed by {trimmed_w}px (x) and {trimmed_h}px (y) "
            f"to fit boxsize={boxsize}.  "
            f"Effective ROI: x=[{x0},{new_x1}], y=[{y0},{new_y1}]  "
            f"({new_width}×{new_height} px, "
            f"{(new_width // boxsize) * (new_height // boxsize)} boxes)."
        )
    else:
        nx = new_width  // boxsize
        ny = new_height // boxsize
        print(
            f"ROI fits boxsize={boxsize} exactly: "
            f"{nx}×{ny} = {nx*ny} boxes."
        )

    return x0, y0, new_x1, new_y1


def extract_boxes(
    image: np.ndarray,
    boxsize: int,
) -> np.ndarray:
    """
    Return a 3-D array of shape (n_boxes, boxsize, boxsize) containing
    every non-overlapping *boxsize*×*boxsize* tile across *image*.
    *image* must already be cropped to the ROI and its dimensions must be
    exact multiples of *boxsize*.
    """
    H, W  = image.shape
    ny, nx = H // boxsize, W // boxsize
    boxes = (image
             .reshape(ny, boxsize, nx, boxsize)
             .transpose(0, 2, 1, 3)
             .reshape(ny * nx, boxsize, boxsize))
    return boxes


# ---------------------------------------------------------------------------
# PTC computation
# ---------------------------------------------------------------------------

def compute_ptc_boxes(
    img_a: np.ndarray,
    img_b: np.ndarray,
    boxsize: int,
    read_noise_var_adu: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Partition the (already ROI-cropped) images into boxes and return per-box
    statistics.

    Returns three 1-D arrays of length n_boxes:
        box_means       = 0.5 * (mean(A_box) + mean(B_box))
        box_total_vars  = 0.5 * var(A_box − B_box)
        box_shot_vars   = box_total_vars − read_noise_var_adu
    """
    boxes_a = extract_boxes(img_a, boxsize)
    boxes_b = extract_boxes(img_b, boxsize)

    diff = boxes_a - boxes_b

    box_means = 0.5 * (boxes_a.mean(axis=(1, 2)) + boxes_b.mean(axis=(1, 2)))

    n_pix = boxsize * boxsize
    diff_flat = diff.reshape(len(diff), n_pix)
    box_total_vars = 0.5 * diff_flat.var(axis=1, ddof=1)

    box_shot_vars = box_total_vars - read_noise_var_adu

    return box_means, box_total_vars, box_shot_vars


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def estimate_read_noise_var(
    means: np.ndarray,
    variances: np.ndarray,
    low_signal_fraction: float = 0.01,
) -> float:
    """
    Estimate read-noise variance in ADU² from the lowest-signal boxes.

    Takes the boxes whose mean falls in the bottom *low_signal_fraction* of
    the observed mean range and returns the median of their variances.  At low
    signal the shot-noise contribution is small and the variance is dominated
    by read noise, so this gives a robust, fit-independent estimate.
    """
    threshold = 1 #means.min() + low_signal_fraction * (means.max() - means.min())  #### EDITED TO USE HARDCODED CUT NOT FRACTION
    low_mask = means <= threshold

    if low_mask.sum() < 1:
        # Fallback: use the single lowest-mean box
        low_mask = means == means.min()

    return float(np.median(variances[low_mask]))


def fit_linear_regime(
    means: np.ndarray,
    variances: np.ndarray,
    linear_fraction: float = 0.35,
) -> Tuple[float, float, np.ndarray]:  # gain, R², linear_mask
    """
    Fit  var = (1/gain) * mean  in the linear regime (slope only; no intercept
    is used to derive detector parameters).

    *means* and *variances* are flat arrays of per-box values across all
    exposure levels and all files.  Linear regime: mean < linear_fraction
    * max(mean).  Returns (gain [e⁻/ADU], R², linear_mask).

    Gain is measured purely from the slope 1/gain = dVar/dMean, independent of
    read noise.  Read-noise variance in ADU² is estimated separately from the
    lowest-signal boxes via estimate_read_noise_var(), and read noise in
    electrons is then  sqrt(read_noise_var_adu) * gain.
    """
    max_mean = means.max()
    mask = means < linear_fraction * max_mean

    if mask.sum() < 2:
        raise RuntimeError("Too few points in the linear regime to fit.")

    slope, _, r_value, _, _ = stats.linregress(
        means[mask], variances[mask]
    )

    if slope <= 0:
        raise RuntimeError(
            f"Fit yielded non-positive slope ({slope:.4g}); "
            "check data or ROI."
        )

    gain = 1.0 / slope

    return gain, r_value ** 2, mask


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_ptc(
    # linearity data: list of (label, exptimes_array, exp_means_array) per file
    linearity_per_file: List[Tuple[str, np.ndarray, np.ndarray]],
    # combined PTC data
    box_means: np.ndarray,
    box_variances: np.ndarray,
    box_shot_variances: np.ndarray,
    gain: float,
    read_noise_var_adu: float,
    read_noise_e: float,
    r2: float,
    linear_mask: np.ndarray,
    roi: Tuple[int, int, int, int],
    boxsize: int,
    n_boxes: int,
    output_path: Optional[Path],
    aduunit: str = 'ADU',
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Photon Transfer Curve Analysis", fontsize=14, fontweight="bold")

    # ── Left: PTC (all files combined) ────────────────────────────────────
    ax = axes[0]

    ax.scatter(
        box_means[~linear_mask], box_variances[~linear_mask],
        color="tomato", edgecolors="darkred", linewidths=0.3, s=10,
        zorder=3, label="Total variance — nonlinear",
    )
    ax.scatter(
        box_means[linear_mask], box_variances[linear_mask],
        color="steelblue", edgecolors="navy", linewidths=0.3, s=10,
        zorder=4, label="Total variance — linear",
    )

    shot_plot = np.where(box_shot_variances > 0, box_shot_variances, np.nan)
    # ax.scatter(
    #     box_means[~linear_mask], shot_plot[~linear_mask],
    #     color="salmon", edgecolors="darkred", linewidths=0.3, s=10, marker="^",
    #     zorder=3, label="Shot noise variance — nonlinear",
    # )
    ax.scatter(
        box_means[linear_mask], shot_plot[linear_mask],
        color="darkorange", edgecolors="saddlebrown", linewidths=0.3, s=10, marker="^",
        zorder=4, label="Shot noise variance — linear",
    )

    # x_fit = np.geomspace(box_means.min() * 0.9, box_means[linear_mask].max() * 1.1, 300)
    # y_total = x_fit / gain + read_noise_var_adu
    # y_shot  = x_fit / gain
    # ax.plot(x_fit, y_total, "k--", linewidth=1.5, label=f"Total fit  (R²={r2:.4f})")
    # ax.plot(x_fit, y_shot,  "k-.", linewidth=1.5, label="Shot noise fit  (slope=1/gain)")
    minmax = np.array([box_means.min(), box_means.max()])
    ax.plot(minmax, minmax/gain**1, 'k--', label="slope=1")

    full_well_e = gain * box_means.max()
    textstr = (
        f"Gain:        {gain:.3f} e⁻/{aduunit}\n"
        f"Read noise:  {read_noise_e:.2f} e⁻\n"
        # f"Read noise var: {read_noise_var_adu:.2f} ADU²\n"
        f"Full well:   {full_well_e:,.0f} e⁻"
    )
    ax.text(
        0.04, 0.96, textstr,
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Mean Signal ({aduunit})")
    ax.set_ylabel(f"Variance ({aduunit}²)")
    ax.set_title("PTC: Variance vs. Mean")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    # ── Right: linearity (one colour per file) ─────────────────────────────
    ax2 = axes[1]

    # Choose colours: one per file
    n_files = len(linearity_per_file)
    if n_files <= 10:
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colours = [palette[i % len(palette)] for i in range(n_files)]
    else:
        cmap = cm.get_cmap("tab20", n_files)
        colours = [cmap(i) for i in range(n_files)]

    all_exptimes = np.concatenate([et for _, et, _ in linearity_per_file])
    all_means    = np.concatenate([em for _, _, em in linearity_per_file])

    for (label, exptimes, exp_means), colour in zip(linearity_per_file, colours):
        i = len(exptimes)//2
        flux = round(sorted(exp_means)[i]/sorted(exptimes)[i],2)
        # flux = round(exp_means[i]/exptimes[i],2)
        ax2.scatter(
            exptimes, exp_means,
            color=colour, edgecolors="black", linewidths=0.4,
            zorder=3, label=f'~{flux} {aduunit}/s',
        )

    # Global linear fit across all files
    # slope_t, intercept_t, r2_t, _, _ = stats.linregress(all_exptimes, all_means)
    # x_t = np.linspace(all_exptimes.min(), all_exptimes.max(), 200)
    # ax2.plot(
    #     x_t, slope_t * x_t + intercept_t, "k--", linewidth=1.5,
    #     label=f"Linear fit  (R²={r2_t:.4f})",
    # )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Exposure Time (s)")
    ax2.set_ylabel(f"Mean Signal ({aduunit})")
    ax2.set_title("Linearity: Signal vs. Exposure Time")
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)

    x0, y0, x1, y1 = roi
    fig.text(
        0.5, 0.01,
        f"ROI: x=[{x0},{x1}], y=[{y0},{y1}]  |  "
        f"boxsize={boxsize}px  |  {n_boxes} boxes per exposure",
        ha="center", fontsize=8, color="gray",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])


    if output_path is not None:
        # output_path.parent.mkdir(parents=True, exist_ok=True) # Make the dir first?
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    # else:
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Photon Transfer Curve (PTC) analysis across one or more "
            "multi-extension FITS files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "fits_pattern",
        nargs="+",
        help=(
            "One or more FITS file paths or a glob pattern "
            "(e.g. 'raw_*.fits').  Quote the pattern to prevent the shell "
            "from expanding it before Python sees it."
        ),
    )
    parser.add_argument(
        "--roi",
        nargs=4, type=int, metavar=("X0", "X1", "Y0", "Y1"),
        required=True,
        help="Region of interest (inclusive pixel coords): X0 X1 Y0 Y1.",
    )
    parser.add_argument(
        "--skip",
        type=int, default=0, metavar="N",
        help=(
            "Ignore the first N extensions (0-indexed) in EACH file. "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--boxsize",
        type=int, default=32, metavar="N",
        help=(
            "Side length in pixels of the square boxes used to sample the ROI. "
            "The ROI is trimmed to the largest area evenly divisible by this size "
            "(default: 32)."
        ),
    )
    parser.add_argument(
        "--linear-fraction",
        type=float, default=0.35, metavar="FRAC",
        help=(
            "Exclude the top (1-FRAC) of the signal range from the linear "
            "fit (default: 0.35)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path, default=Path("./"), metavar="DIR",
        help=(
            "Directory in which to save the output PNG plot "
            "(default: current directory)"
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path, default=None, metavar="DIR",
        help=(
            "Directory in which to save a per-exposure mean/variance CSV. "
        ),
    )

    args = parser.parse_args()

    # ── Expand glob patterns into a sorted, deduplicated file list ──────────
    fits_files: List[Path] = []
    seen = set()
    for pattern in args.fits_pattern:
        matched = sorted(glob.glob(pattern))
        if not matched:
            # Treat as a literal path so the user gets a clear error
            matched = [pattern]
        for p in matched:
            rp = Path(p).resolve()
            if rp not in seen:
                seen.add(rp)
                fits_files.append(Path(p))

    if not fits_files:
        sys.exit("Error: no FITS files matched the supplied pattern(s).")

    for fp in fits_files:
        if not fp.is_file():
            sys.exit(f"Error: {fp} does not exist or is not a file.")

    # Check for multiple gains
    if ('hg.fits' in str(fits_files).lower()) and ('lg.fits' in str(fits_files).lower()):
        sys.exit(f"Error: filenames indicate both high and low gain (hg, lg).  Cannot combine these.")

    print(f"Files to process ({len(fits_files)}):")
    for fp in fits_files:
        print(f"  {fp}")
    print()

    if args.boxsize < 1:
        sys.exit("Error: --boxsize must be >= 1.")

    roixy = parse_roi(args.roi)
    print(
        f"Requested ROI: x=[{roixy[0]},{roixy[2]}], y=[{roixy[1]},{roixy[3]}]  "
        f"({roixy[2]-roixy[0]+1}×{roixy[3]-roixy[1]+1} px)"
    )
    roixy = trim_roi_to_boxsize(roixy, args.boxsize)
    x0, y0, x1, y1 = roixy
    n_boxes = ((x1 - x0 + 1) // args.boxsize) * ((y1 - y0 + 1) // args.boxsize)
    roi = (slice(y0,y1+1), slice(x0,x1+1))
    print()

    # ── Per-file processing ──────────────────────────────────────────────────
    # Combined arrays (all files) for the PTC fit
    all_box_means      = []
    all_box_total_vars = []

    # Per-file data for the linearity plot and second pass
    per_file_records = []   # list of dicts, one per file

    # Check for consistent gains
    adukeys = []
    try:
        for fits_path in fits_files:
            with fits.open(fits_path) as temphdulist:
                for hdu in temphdulist[1:]: # Skip primary
                    adukeys.append(hdu.header[ADUKEY])
    except:
        print(f"WARNING: Could not find {ADUKEY} header for every image.  Units will be ADUf.")
        adukeys = []

    unique_adukeys = len(set(adukeys))
    if unique_adukeys == 0:     # Caught missing keys exception --> don't use
        aduunit = 'ADUf'
        adu_convert = 1
    elif unique_adukeys > 1:    # Multiple keys --> stop
        printf(f"ERROR: Mismatched {ADUKEY} headers:",str(adukeys))
        sys.exit(1)
    else:                       # All keys same --> use it
        aduunit = 'ADUe'
        adu_convert = adukeys[0]
    # end check


    for fits_path in fits_files:
        extensions = load_extensions(fits_path, roi=roi, skip=args.skip, adu_convert=adu_convert)
        if not extensions:
            continue

        pairs = group_into_pairs(extensions, filename=fits_path.name)
        if not pairs:
            print(f"  No valid pairs in {fits_path.name}, skipping.\n")
            continue

        print(f"  Found {len(pairs)} exposure-time pair(s) in {fits_path.name}.")

        file_box_means      = []
        file_box_total_vars = []
        file_exp_records    = []   # (exptime_s, median_mean)
        file_pair_images    = {}

        for exptime in sorted(pairs):
            img_a, img_b = pairs[exptime]
            try:
                bm, bv, _ = compute_ptc_boxes(img_a, img_b, args.boxsize)
                file_box_means.append(bm)
                file_box_total_vars.append(bv)
                file_exp_records.append((exptime, float(np.median(bm))))
                file_pair_images[exptime] = (img_a, img_b)
                print(
                    f"    {TIMEKEY}={exptime:10.4f} s  |  "
                    f"median mean={np.median(bm):10.2f} ADU  |  "
                    f"median total var={np.median(bv):12.2f} ADU²  "
                    f"({n_boxes} boxes)"
                )
            except Exception as exc:
                print(f"    {TIMEKEY}={exptime:10.4f} s  →  ERROR: {exc}")

        if not file_exp_records:
            print(f"  No usable pairs computed in {fits_path.name}, skipping.\n")
            continue

        all_box_means.extend(file_box_means)
        all_box_total_vars.extend(file_box_total_vars)

        per_file_records.append({
            "path":        fits_path,
            "box_means":   file_box_means,
            "box_vars":    file_box_total_vars,
            "exp_records": file_exp_records,
            "pair_images": file_pair_images,
        })
        print()

    if not per_file_records:
        sys.exit("Error: no usable data found across all files.")

    # Derive output filenames from the first FITS file alphabetically
    first_fits = sorted(fits_files)[0]
    base_stem = first_fits.stem
    png_path = args.output / (base_stem + ".png")
    csv_path = (args.csv / (base_stem + ".csv")) if args.csv else None

    total_exposures = sum(len(r["exp_records"]) for r in per_file_records)
    if total_exposures < 3:
        sys.exit("Error: need at least 3 valid exposure levels in total for a PTC.")

    # ── Combine all boxes for the global PTC fit ─────────────────────────────
    box_means      = np.concatenate(all_box_means)
    box_total_vars = np.concatenate(all_box_total_vars)

    # ── Fit over all boxes ───────────────────────────────────────────────────
    try:
        gain, r2, linear_mask = fit_linear_regime(
            box_means, box_total_vars,
            linear_fraction=args.linear_fraction,
        )
    except RuntimeError as exc:
        sys.exit(f"Fit failed: {exc}")

    # ── Read noise: median variance of lowest-signal boxes (ADU²) ────────────
    # Empirical floor of the variance at low signal, independent of the PTC
    # slope fit.  Read noise in electrons follows from the gain.
    read_noise_var_adu = estimate_read_noise_var(box_means, box_total_vars)
    read_noise_e       = np.sqrt(read_noise_var_adu) * gain

    # ── Second pass: recompute shot variance with read noise subtracted ───────
    all_box_shot_vars  = []
    for rec in per_file_records:
        for exptime, (img_a, img_b) in sorted(rec["pair_images"].items()):
            _, _, bsv = compute_ptc_boxes(
                img_a, img_b, args.boxsize, read_noise_var_adu
            )
            all_box_shot_vars.append(bsv)
    box_shot_vars = np.concatenate(all_box_shot_vars)

    full_well_e = gain * box_means.max()

    print(f"\n{'='*52}")
    print(f"  Gain            : {gain:.4f} e⁻/ADU")
    print(f"  Read noise      : {read_noise_e:.3f} e⁻")
    print(f"  Read noise var  : {read_noise_var_adu:.3f} ADU²")
    print(f"  Full well       : {full_well_e:,.0f} e⁻  (at max measured signal)")
    print(f"  Linear fit R²   : {r2:.6f}  "
          f"({linear_mask.sum()}/{len(box_means)} boxes used)")
    print(f"{'='*52}\n")

    # ── Optional CSV ─────────────────────────────────────────────────────────────────────────────
    if csv_path is not None:
        import csv
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        exp_index  = []   # exptime label per box
        file_index = []   # filename label per box
        for rec in per_file_records:
            fname = rec["path"].name
            for exptime in sorted(rec["pair_images"]):
                exp_index.extend([exptime] * n_boxes)
                file_index.extend([fname]  * n_boxes)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename", "exptime_s", "box_index", "mean_ADU",
                "total_variance_ADU2", "shot_variance_ADU2", "in_linear_regime",
            ])
            for i, (fn, et, m, tv, sv, lm) in enumerate(zip(
                file_index, exp_index, box_means, box_total_vars,
                box_shot_vars, linear_mask,
            )):
                writer.writerow([
                    fn, et, i % n_boxes,
                    f"{m:.4f}", f"{tv:.4f}", f"{sv:.4f}", lm,
                ])
        print(f"CSV saved to {csv_path}")

    # ── Build per-file linearity data for the plot ────────────────────────────
    linearity_per_file = []
    for rec in per_file_records:
        exp_records = sorted(rec["exp_records"], key=lambda r: r[1])
        exptimes  = np.array([r[0] for r in exp_records])
        exp_means = np.array([r[1] for r in exp_records])
        linearity_per_file.append((rec["path"].name, exptimes, exp_means))

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_ptc(
        linearity_per_file,
        box_means, box_total_vars, box_shot_vars,
        gain, read_noise_var_adu, read_noise_e, r2, linear_mask,
        roixy, args.boxsize, n_boxes, png_path, aduunit
    )


if __name__ == "__main__":
    main()