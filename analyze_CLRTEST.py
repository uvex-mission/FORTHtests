import argparse
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import RCtools

TIMEKEY = 'EXPTIME'
TGKEY = 'VHIGH_TG'
t_dark = 2.0 # TIMEKEY value for the short dark exposures
SIGMACLIP = 3.

LEDVlist = [1.945, 2.363, 3.165]  # LED voltages for each FITS file

# VHIGH_TG_list = [1.8, 1.9, 2.0, 2.1, 2.2, 2.3] # Hardcoded order in single CLRTESTx file
N_CLRTEST = 240 # Expected number of extensions in single CLRTESTx file
N_LAG = 4  # number of extensions after the flash to use for lag measurements

##########################################

def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze CLR test FITS files and plot lag vs. stimulus.'
    )
    parser.add_argument(
        'files',
        nargs='+',
        metavar='FILE_OR_GLOB',
        help=(
            'Three CLR test FITS files (clrtest1, clrtest2, clrtest3) in order, '
            'or a single glob/regex pattern that resolves to those three files. '
            'A single CSV file may also be provided to skip directly to plotting.'
        )
    )
    parser.add_argument(
        '--roi',
        nargs=4,
        type=int,
        default=None,
        metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
        help=(
            'Analysis region in pixel coordinates: XMIN XMAX YMIN YMAX (column and row bounds). '
            'Required when processing FITS files; ignored (with a warning) when loading a CSV.'
        )
    )
    parser.add_argument(
        '--outdir',
        default='./',
        metavar='DIR',
        help='Output directory for CSV and PNG files. Defaults to ./'
    )
    return parser.parse_args()


def resolve_files(file_args):
    """Expand globs/patterns and return a sorted list of matched paths."""
    resolved = []
    for pattern in file_args:
        matches = sorted(glob.glob(pattern))
        if matches:
            resolved.extend(matches)
        else:
            # Treat as a literal path
            resolved.append(pattern)
    return resolved


def find_clrtest_files(file_args):
    """
    Accept either:
      - A single CSV file  → return it as-is.
      - Three explicit FITS paths or a glob that resolves to three FITS files
        containing 'clrtest1', 'clrtest2', 'clrtest3' → return them sorted by index.
    """
    paths = resolve_files(file_args)

    # Single CSV shortcut
    if len(paths) == 1 and paths[0].lower().endswith('.csv'):
        return paths  # Caller will handle this case

    # Filter to files whose basename contains clrtestN
    clrtest_re = re.compile(r'clrtest([123])', re.IGNORECASE)
    tagged = []
    for p in paths:
        m = clrtest_re.search(os.path.basename(p))
        if m:
            tagged.append((int(m.group(1)), p))

    if len(tagged) != 3:
        sys.exit(
            f'ERROR: Expected exactly 3 files matching clrtest1/2/3, '
            f'got {len(tagged)}: {[p for _, p in tagged]}'
        )

    tagged.sort(key=lambda x: x[0])
    return [p for _, p in tagged]


def make_output_stem(clrtest1_path, outdir):
    """Return the output path stem (no extension) derived from the clrtest1 filename."""
    basename = os.path.splitext(os.path.basename(clrtest1_path))[0]
    directory = outdir if outdir else os.path.dirname(os.path.abspath(clrtest1_path))
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, basename)


def main():
    args = parse_args()

    files = find_clrtest_files(args.files)
    is_csv = len(files) == 1 and files[0].lower().endswith('.csv')

    # Validate --roi presence depending on input type
    if is_csv:
        if args.roi is not None:
            print('WARNING: --roi is ignored when loading from a CSV file.')
        roi = None
    else:
        if args.roi is None:
            sys.exit('ERROR: --roi XMIN XMAX YMIN YMAX is required when processing FITS files.')
        # --roi XMIN XMAX YMIN YMAX  →  array[rows, cols] = array[YMIN:YMAX, XMIN:XMAX]
        xmin, xmax, ymin, ymax = args.roi
        roi   = (slice(ymin, ymax), slice(xmin, xmax))

    # ── CSV shortcut: skip data reduction, go straight to plotting ──────────
    if is_csv:
        csv_path = files[0]
        print(f'Loading existing CSV: {csv_path}')
        df = pd.read_csv(csv_path)
        stem = make_output_stem(csv_path, args.outdir)
    else:
        # ── Full reduction from FITS ─────────────────────────────────────────
        flist = files
        stem  = make_output_stem(flist[0], args.outdir)

        dflist = []
        for f, ledv in zip(flist, LEDVlist):
            dfi = RCtools.all_headers_to_df(f)
            assert len(dfi) == N_CLRTEST,               f'{f}: expected {N_CLRTEST} extensions, got {len(dfi)}'
            # assert N_CLRTEST % len(VHIGH_TG_list) == 0, 'N_CLRTEST must be divisible by number of VHIGH_TG values'
            # NVrepeat = N_CLRTEST // len(VHIGH_TG_list)
            # dfi['VHIGH_TG'] = np.tile(VHIGH_TG_list, (NVrepeat, 1)).T.flatten()
            dfi['LEDV'] = ledv
            dflist.append(dfi)

        df = pd.concat(dflist).reset_index()
        del dflist

        header_list = ['FILENAME', 'EXTN', 'EXTNAME', TIMEKEY, 'GAINFITS', 'LEDV', TGKEY]
        df = df[header_list]

        # Identify extensions with LED flash and subsequent darks
        df['FLASH'] = df[TIMEKEY] > 2.0  ### Identify flashes as the longer exposures
        i_flash = df.index[df['FLASH'] == True]

        flashi = np.array([-1]*len(df))
        for ii in range(N_LAG+1):
            flashi[i_flash + ii] = ii
        df['FLASHI'] = flashi  # FLASHI column is 0 for each flash and counts up

        # Convert VHIGH_TG to volts
        df[TGKEY] = round( RCtools.DAC_to_V(df[TGKEY]) , 3)

        print('Processing images...')
        rows = df.progress_apply(RCtools.compute_cds_stats,
                        axis=1, clip_sigma=SIGMACLIP, do_median=False, roi=roi).tolist()

        df = pd.concat([df, pd.DataFrame(rows)], axis=1)

        # ── Save CSV ─────────────────────────────────────────────────────────
        outcsv = stem + '.csv'
        df.to_csv(outcsv, index=False)
        print(f'Saved: {outcsv}')

    # ── Plot Lag vs. stimulus for each VHIGH_TG ──────────────────────────────
    outpng = stem + '_lag0.png'

    plt.figure(figsize=(5, 10))

    for v in df[TGKEY].unique():
        df_v = df[df[TGKEY] == v].reset_index()

        i_flash = df_v.index[df_v['FLASH'] == True]
        x = df_v.loc[i_flash]['mean']       # images with flashes
        y = df_v.loc[i_flash + 1]['mean']   # images just after flashes
        z = df_v.loc[i_flash - 1]['mean']   # images just before flashes

        plt.scatter(x.values - z.values, y.values - z.values, label=f'{v}', s=10)

    plt.xscale('log')
    plt.xlabel('Stimulus (ADU)')
    plt.ylabel('Lag_0 (ADU)')
    plt.legend(title=f'{TGKEY}')
    plt.savefig(outpng)
    # plt.show()
    print(f'Saved: {outpng}')


    # ── Plot Lag vs. frame for each VHIGH_TG, highest stimulus only ──────────────────────────────
    outpng = stem + '_lagt.png'

    # plt.figure(figsize=(5, 10))
    plt.figure()
    plt.title('Timeseries after flash (Frame 0)')

    for v in df[TGKEY].unique():
        df_v = df[df[TGKEY] == v].reset_index()

        i_flash = df_v.index[(df_v[TIMEKEY]==df_v[TIMEKEY].max()) * (df_v['LEDV']==df_v['LEDV'].max()) ]  # Brightest flash
        i_flash = i_flash[0] # Should only have 1 element

        y = df_v.loc[i_flash:i_flash+N_LAG]['mean']  # series after flash
        z = df_v.loc[i_flash - 1]['mean']   # images just before flashes

        # breakpoint()

        plt.scatter(range(len(y)), y.values - z, label=f'{v}', s=10)

    plt.yscale('log')
    plt.xlabel('Frame #')
    plt.ylabel('ROI mean (ADU)')
    plt.legend(title=f'{TGKEY}')
    plt.savefig(outpng)
    # plt.show()
    print(f'Saved: {outpng}')


if __name__ == '__main__':
    main()
