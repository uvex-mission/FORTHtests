"""
analyze_parmchk.py

For each image extension in a FITS file, computes mean, median (optional), and
standard deviation of:
  - the baseline frame  (data[0, :, :])

Results are collected into a pandas DataFrame and printed to stdout.
Optionally saves the table to a CSV file.

Usage:
    python analyze_parmchk.py <path_to_fits_file> [--csv output.csv]
                               [--sigclip SIGMA] [--median] [--plot]

Examples:
    python analyze_parmchk.py data.fits
    python analyze_parmchk.py data.fits --csv results.csv
"""

import argparse
import astropy.io.fits as pf
import numpy as np
import pandas as pd
import sys

from RCtools import *

# --------------------------------------------------------------------------- #
# Global constants
# --------------------------------------------------------------------------- #
DETSIZE = 4096
CHANSIZE = 256

V_MAX = 3.3

# PASS/FAIL choices
MEANoverSIG_IDEAL = 5.7 # diagnostic; not used in PASS/FAIL
MEANoverSIG = (4,7) # PASS/FAIL range
LOWTHRESH = 4  # number of sigma to require 0 ADU below 3.3V

MAX_SUBDIVISIONS = 2  # Maximum number of times to quarter the column range on retry # --> 256 cols min

### Hardcoded order of tests in the raw PARMCHK file
# All are high gain except those marked LG
test_labels = [
    "HG-TG*", "HG-TG*", "HG-TG*",
    "HG-TG",  "HG-TG",  "HG-TG",
    "glow",   "glow",   "glow",
    "testen", "testen",  # 9 and 10
    "LG-TG*", "LG-TG*", "LG-TG*",
    "LG-TG",  "LG-TG",  "LG-TG",
]
i_glow = 6     # 1st glow test 
i_testen = 10  # Use for reference voltage
i_lowgain = 11 # 1st low gain image

HLINE = 60

def col_to_chan(col: int) -> int:
    """Return the channel number for a given column index."""
    return col // CHANSIZE

def chan_range_str(col_start: int, col_end: int) -> str:
    """Return a string like 'cols 0–1023  [ch 0–3]'."""
    ch_start = col_to_chan(col_start)
    ch_end   = col_to_chan(col_end - 1)
    ch_str   = f"ch {ch_start}" if ch_start == ch_end else f"ch {ch_start}–{ch_end}"
    return f"cols {col_start}–{col_end - 1}  [{ch_str}]"

def analyze_fits(fits_path: str, clip_sigma: float | None,
                 do_median: bool, roi=None) -> pd.DataFrame:

    # Start with table of metadata for easy organization
    df_all = all_headers_to_df(fits_path)

    e_per_ADU = df_all['GAINFITS']
    KSCALE = df_all['KSCALE']  # Factor applied to FITS file to reduce file size
    # TRY: GAIN_F2E = df_all['GAIN_F2E']

    e_per_ADU[i_lowgain:] *= GAIN_LO_o_HI      # Adjust Low gain values
    V_per_ADU = ADU_to_V(df_all['PIXELT'], df_all['SETTLET'], df_all['LSB_DROP']) *KSCALE
    # V_per_e = V_per_ADU / e_per_ADU

    # List of dicts with computations
    rows = df_all.apply(compute_baseline_stats, axis=1, clip_sigma=clip_sigma, do_median=do_median, roi=roi).tolist()

    # Merge results with some metadata from original table
    df = pd.concat([df_all[['EXTNAME','PIXELT','SETTLET','LSB_DROP','GAINFITS','V_EXTRA']], pd.DataFrame(rows)], axis=1)

    df.insert(0, "TEST", test_labels)  # Put TEST column first after index

    # Compute extra columns for deciding PASS/FAIL

    # Baseline must be low enough to allow OK dynamic range
    # This is Rick's rule of thumb
    MoS = df['mean']/df['std']
    pf = passfail( (MEANoverSIG[0]<=MoS) * (MoS<=MEANoverSIG[1])  )
    pf[i_glow:i_lowgain] = '-' # PASS/FAIL makes no sense for these rows
    df['basetest1'] = pf

    df['std_thresh_V'] = MEANoverSIG_IDEAL*df['std']*V_per_ADU
    df['maxout_V'] = V_MAX - df['std_thresh_V']

    df['V_EXTRA'] = DAC_to_V(df['V_EXTRA'])  # Convert V_EXTRA DAC setting to Volts
    df.insert(len(df.columns)-1, 'V_EXTRA', df.pop('V_EXTRA'))  # move column to last (so far)

    baseline_V = (df['mean'] - df['mean'].iloc[i_testen]) * V_per_ADU
    df['mean (V)'] = df['V_EXTRA'] - baseline_V
    
    # Baseline V should be sufficiently far from V_MAX rail or ADC will miss low e- values
    clearance = (V_MAX - df['mean (V)'])/(df['std']*V_per_ADU)
    df[f'({V_MAX}V-mean)/std'] = clearance

    pf = passfail( (clearance >= LOWTHRESH) * np.isfinite(clearance))  # Meets threshold but not Inf!
    pf[i_glow:i_lowgain] = '-' # PASS/FAIL makes no sense for these rows
    df['basetest2'] = pf

    # Round before adding the tiny numbers
    df = df.round(6)

    df['e/ADU'] = e_per_ADU
    df['V/ADU'] = V_per_ADU
    # df['V/e'] = V_per_e

    # Don't need these in the results table
    del df['PIXELT'], df['SETTLET'], df['LSB_DROP'], df['GAINFITS']

    return df


def check_pass(df: pd.DataFrame) -> bool:
    """Return True if the dataframe contains no FAIL entries in either test column."""
    return not (
        (df['basetest1'] == 'FAIL').any() or
        (df['basetest2'] == 'FAIL').any()
    )


def prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply column renames and significant-figure formatting for display/CSV."""
    renames = {
        'basetest1': f"{MEANoverSIG[0]}<μ/σ<{MEANoverSIG[1]}",
        'basetest2': f'>{LOWTHRESH}',
        'std_thresh_V': f'std*{MEANoverSIG_IDEAL} (V)',
        'maxout_V': f'maxOutput (V)',
    }
    df = df.rename(columns=renames)
    df = df.map(custom_sig_figs)
    return df


def print_region_result(df: pd.DataFrame, col_start: int, col_end: int,
                        passed: bool, label: str = "") -> None:
    """Print a labelled results table for one column region."""
    header = chan_range_str(col_start, col_end)
    if label:
        header += f"  [{label}]"
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    print('Baseline calculations (from 1st frame in CDS pair):\n')
    print(df.to_string(), '\n')
    print('Test result:', passfail(passed))
    print()


def main():
    parser = argparse.ArgumentParser(description="FITS multi-extension statistics")
    parser.add_argument("fits_file", help="Path to the input FITS file")
    parser.add_argument(
        "--csv", metavar="OUTPUT_CSV",
        help="Optional path to save the results as a CSV file",
    )
    parser.add_argument(
        "--sigclip", metavar="SIGMA", type=float, default=3.0,
        help="Sigma-clip data before computing statistics (Default=3σ)",
    )
    parser.add_argument(
        "--median", action="store_true", default=False,
        help="Include median columns in the output (adds runtime; omitted by default).",
    )

    args = parser.parse_args()

    print()
    print(f"Opening {args.fits_file} …\n")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    # ------------------------------------------------------------------ #
    # Full-detector pass
    # ------------------------------------------------------------------ #
    roi = (slice(None), slice(0, DETSIZE))
    df_full = analyze_fits(args.fits_file, clip_sigma=args.sigclip,
                           do_median=args.median, roi=roi)
    passed_full = check_pass(df_full)

    csv_rows = []  # accumulate (header_label, display_df) pairs for CSV

    disp = prepare_display_df(df_full.copy())
    print_region_result(disp, 0, DETSIZE, passed_full, label="full detector")
    csv_rows.append((f"cols_0-{DETSIZE-1}_ch_0-{col_to_chan(DETSIZE-1)}", disp))

    if passed_full:
        overall_pass = True
    else:
        # ------------------------------------------------------------------ #
        # Iterative column-halving.
        # Queue entries: (col_start, col_end, depth)
        # A region is re-queued for further splitting only if it fails
        # and depth < MAX_SUBDIVISIONS.
        # ------------------------------------------------------------------ #
        print("=" * HLINE)
        print(f"Full-detector FAIL — subdividing into quarters "
            f"(up to {MAX_SUBDIVISIONS} levels)\n")

        # Track per-region outcomes to determine the overall summary
        region_results: list[tuple[int, int, bool]] = []

        queue: list[tuple[int, int, int]] = [(0, DETSIZE, 1)]  # (col_start, col_end, depth)

        while queue:
            col_start, col_end, depth = queue.pop(0)

            col_q = (col_end - col_start) // 4
            boundaries = [col_start + i * col_q for i in range(5)]
            # Absorb any rounding remainder into the last segment
            boundaries[-1] = col_end
            
            for seg_start, seg_end in zip(boundaries, boundaries[1:]):

                roi = (slice(None), slice(seg_start, seg_end))
                df_seg = analyze_fits(args.fits_file, clip_sigma=args.sigclip,
                                      do_median=args.median, roi=roi)
                passed_seg = check_pass(df_seg)

                disp = prepare_display_df(df_seg.copy())
                label = f"level {depth}/{MAX_SUBDIVISIONS}"
                print_region_result(disp, seg_start, seg_end, passed_seg, label=label)
                csv_rows.append((f"cols_{seg_start}-{seg_end-1}_ch_{col_to_chan(seg_start)}-{col_to_chan(seg_end-1)}", disp))

                if passed_seg:
                    region_results.append((seg_start, seg_end, True))
                elif depth < MAX_SUBDIVISIONS:
                    queue.append((seg_start, seg_end, depth + 1))
                else:
                    # Max depth reached and still failing
                    region_results.append((seg_start, seg_end, False))

        overall_pass = any(passed for _, _, passed in region_results)

        # Summary of leaf regions — sorted by column start
        print("=" * HLINE)
        print("SUBDIVISION SUMMARY")
        print("=" * HLINE)
        for seg_start, seg_end, passed in sorted(region_results, key=lambda r: r[0]):
            print(f"  {chan_range_str(seg_start, seg_end)} : {passfail(passed)}")
        print()

    print(f"Overall test result (are any regions OK?): {passfail(overall_pass)}")

    # ------------------------------------------------------------------ #
    # CSV output — one section per region, separated by a blank row
    # ------------------------------------------------------------------ #
    if args.csv:
        with open(args.csv, 'w') as f:
            for section_label, disp_df in csv_rows:
                f.write(f"{section_label}\n")
                disp_df.to_csv(f)
                f.write("\n")
        print(f"\nResults saved to {args.csv}")


if __name__ == "__main__":
    main()
