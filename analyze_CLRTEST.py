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
SIGMACLIP = 3.
HISTMAX_QUANTILE = 99.9  # Used to limit x-range of lag histograms

LEDVlist = [1.945, 2.363, 3.165]  # LED voltages for each FITS file
# LEDVlist = [2.4, 2.4, 2.4]  # LED voltages for each FITS file


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

    # files = find_clrtest_files(args.files) # Require clrtest 1-2-3 pattern
    files = resolve_files(args.files)
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
            dfi['LEDV'] = ledv
            dflist.append(dfi)

        df = pd.concat(dflist).reset_index()
        del dflist

        header_list = ['FILENAME', 'EXTN', 'EXTNAME', TIMEKEY, 'GAINFITS', 'LEDV', TGKEY]
        df = df[header_list]

        # Round the exposure times to avoid lots of unique values from ms imprecision
        df[TIMEKEY] = df[TIMEKEY].round(2)

        # Convert VHIGH_TG to volts
        df[TGKEY] = round( RCtools.DAC_to_V(df[TGKEY]) , 2)

        print('Processing images...')
        gainfits = True
        rows = df.progress_apply(RCtools.compute_cds_stats,
                        axis=1, clip_sigma=SIGMACLIP, do_median=False, quantiles=[HISTMAX_QUANTILE], roi=roi, gainfits=gainfits).tolist() ### HARDCODED QUANTILE

        df['units'] = 'e-' if gainfits else 'ADUf'
        df = pd.concat([df, pd.DataFrame(rows)], axis=1)


        # ── Save CSV ─────────────────────────────────────────────────────────
        outcsv = stem + '.csv'
        df.to_csv(outcsv, index=False)
        print(f'Saved: {outcsv}')

    # -- Identify the flash / dark pattern ------------------------------

    # Guess the dark exposure time from the last image
    t_dark = df[TIMEKEY].iloc[-1]

    # Identify extensions with LED flash and subsequent darks
    df['FLASH'] = (df[TIMEKEY] != t_dark) * (df[TIMEKEY]>0)  ### WARNING: Skips the flash if it has same exptime as darks
    i_flash = df.index[df['FLASH'] == True]

    # Guess the number of lag measurements (darks) from the last flash index
    N_LAG = max(df.index) - max(i_flash)

    flashi = np.array([-1]*len(df))
    for ii in range(N_LAG+1):
        flashi[i_flash + ii] = ii
    df['FLASHI'] = flashi  # FLASHI column is 0 for each flash and counts up


    print('Starting plots...')

    # Identify units for plots
    units = df['units'].iloc[0]  # Assumes all units are same

    # ── Timeseries of ROI mean for each VHIGH_TG ──────────────────────────────

    outpng = stem + '_tseries.png'

    vlist = df['LEDV'].unique()
    
    fig, axes = plt.subplots(nrows=len(vlist), sharex=True, 
                            figsize=(8, 2.5*len(vlist))) # scale figure to number of bias voltages

    # Convert into a 1-element list if only 1 axis
    axlist = [axes] if isinstance(axes, plt.Axes) else axes
    ax = axlist[0]
    ax.text(1.02, 1.01, 'LEDV', transform=ax.transAxes)
    ax.text(.05, 1.01, TGKEY, transform=ax.transAxes)

    for ax,v in zip(axlist,vlist):
        df_v = df[df['LEDV'] == v].reset_index()

        ax.plot(df_v['mean'], label=f'{v}')
        # ax.legend(title=TGKEY, loc='center left')
        ax.text(1.02, 0.5, str(v), transform=ax.transAxes)
        ax.set_ylabel(f'ROI mean ({units})')

        # VHIGH_TG labels aligned to x positions
        for tg in df_v[TGKEY].unique():
            x = df_v[df_v[TGKEY]==tg].index.start  # first index with this VHIGH_TG
            ax.text(x, df_v['mean'].max()*0.9, f'{tg}')

    plt.xlabel('Frame Number')
    # plt.tight_layout()
    plt.savefig(outpng)
    # plt.show()
    print('Saved '+outpng)

    # ── Plot 1st Lag vs. stimulus for each VHIGH_TG ──────────────────────────────

    markersize=4
    marker='s'

    outpng = stem + '_lag1.png'

    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(8, 8)) 

    for v in df[TGKEY].unique():
        df_v = df[df[TGKEY] == v].reset_index()

        i_flash = df_v.index[df_v['FLASH'] == True]
        x = df_v.loc[i_flash]['mean'].values       # images with flashes
        y = df_v.loc[i_flash + 1]['mean'].values   # images just after flashes
        z = df_v.loc[i_flash - 1]['mean'].values   # images just before flashes

        ### HARDCODE
        # z = z*0

        axes[0].plot(x-z, y-z, label=f'{v}', 
                    markersize=markersize, marker=marker)

        axes[1].plot(x-z, (y-z)/(x-z), label=f'{v}', 
                    markersize=markersize, marker=marker)

    plt.xscale('log')
    axes[1].set_xlabel(f'Stimulus ({units})')
    axes[0].set_ylabel(f'Lag_1 ({units})')
    axes[1].set_ylabel(f'Lag_1 fraction')
    axes[0].legend(title=f'{TGKEY}')
    axes[1].legend(title=f'{TGKEY}')

    ### HARDCODED LIMITS
    axes[0].set_xlim(10,1.5E5)
    axes[0].set_ylim(-10,1000)  # Absolute
    axes[1].set_xlim(10,1.5E5)
    axes[1].set_ylim(-.01,.3)   # Fraction

    plt.tight_layout()
    plt.savefig(outpng)
    # plt.show()
    print('Saved '+outpng)


    # ── Plot Lag vs. frame for each VHIGH_TG, highest stimulus only ──────────────────────────────
    outpng = stem + '_lagt.png'

    fig, ax = plt.subplots()
    plt.title('Timeseries after flash (Frame 0)')

    df_max = df[df['LEDV']==LEDVlist[1]].reset_index()  # Brightest high-gain LED setting
    # df_max = df[df['LEDV']==max(LEDVlist)].reset_index()  # Brightest LED setting

    for v in df_max[TGKEY].unique():
        df_v = df_max[df_max[TGKEY] == v].reset_index()

        i_flash = df_v.index[df_v[TIMEKEY]==df_v[TIMEKEY].max()]  # Brightest flash
        i_flash = i_flash[0] # Should only have 1 element

        y = df_v.loc[i_flash:i_flash+N_LAG]['mean'].values  # series after flash
        z = df_v.loc[i_flash - 1]['mean']   # images just before flashes

        ### HARDCODE for dark subtraction test
        # z = z*0
        w = y-z

        ax.plot(range(len(y)), w, label=f'{v}',
                    markersize=markersize, marker=marker)

    ax.set_yscale('log')
    ax.set_xlabel('Frame #')
    ax.set_ylabel(f'ROI mean ({units})')

    # Add a second x-axis for exposure time
    secax = ax.secondary_xaxis('bottom', functions=(lambda x: x*t_dark, lambda x: x/t_dark))
    secax.set_xlabel("Total elapsed time (s)")
    secax.spines['bottom'].set_position(('outward', 40)) # push the 2nd axis down to prevent overlap

    plt.legend(title=f'{TGKEY}')
    plt.tight_layout()
    plt.savefig(outpng)
    print('Saved '+outpng)


    # ── Plot CUMULATIVE Lag vs. frame for each VHIGH_TG, highest stimulus only ──────────────────────────────
    outpng = stem + '_lagt_sum.png'

    fig, ax = plt.subplots()
    plt.title('Cumulative lag after flash (Frame 0)')

    df_max = df[df['LEDV']==LEDVlist[1]].reset_index()  # Brightest high-gain LED setting
    # df_max = df[df['LEDV']==max(LEDVlist)].reset_index()  # Brightest LED setting

    for v in df_max[TGKEY].unique():
        df_v = df_max[df_max[TGKEY] == v].reset_index()

        i_flash = df_v.index[df_v[TIMEKEY]==df_v[TIMEKEY].max()]  # Brightest flash
        i_flash = i_flash[0] # Should only have 1 element

        y = df_v.loc[i_flash:i_flash+N_LAG]['mean'].values  # series after flash
        z = df_v.loc[i_flash - 1]['mean']   # images just before flashes

        ### HARDCODE for dark subtraction test
        # z = z*0
        w = y-z

        ax.plot(range(len(y))[1:], np.cumsum(w[1:]), label=f'{v}',
                    markersize=markersize, marker=marker)

    # plt.yscale('log')
    ax.set_xlabel('Frame #')
    ax.set_ylabel(f'ROI mean ({units})')

    # Add a second x-axis for exposure time
    secax = ax.secondary_xaxis('bottom', functions=(lambda x: x*t_dark, lambda x: x/t_dark))
    secax.set_xlabel("Total integration time (s)")
    secax.spines['bottom'].set_position(('outward', 40)) # push the 2nd axis down to prevent overlap

    plt.legend(title=f'{TGKEY}')
    plt.tight_layout()
    plt.savefig(outpng)
    # plt.show()
    print('Saved '+outpng)


    # -- Plot histograms for all VHIGH_TG ------------------------------------
    print('Plotting histograms...')

    df_max = df[df['LEDV']==LEDVlist[1]].reset_index()  # Brightest high-gain LED setting
    # df_max = df[df['LEDV']==max(LEDVlist)].reset_index()  # Brightest LED setting

    i_flash = df_max.index[(df_max[TIMEKEY]==df_max[TIMEKEY].max())]  # List of max flashes, 1 for each VHIGH_TG

    ### HARDCODES

    roi = (slice(10,410), slice(256*2,256*3))
    BINSTEP = 1
    gainfits = True
    units = 'e-' if gainfits else 'ADUf'
    MAXMARGIN = 3  # Histogram upper limit -- multiplies max mean lag of each VHIGH_TG

    for ii in np.arange(N_LAG)+1:  # count frames after each flash

        outpng = stem + f'_hist{ii}.png'
        df_lag = df_max.loc[i_flash+ii] # iith row after the flashes 

        kgain = df_lag['GAINFITS'].iloc[0]

        # binmax = int( df_max[df_max['FLASHI']==ii]['mean'].max() * MAXMARGIN )
        binmax = df_lag[str(HISTMAX_QUANTILE)].max()
        bins = np.arange(0,binmax,BINSTEP*kgain)  # Scale BINSTEP by kgain to avoid weird rounding effects in plots
        x = (bins[1:]+bins[:-1])/2

        plt.figure()
        plt.title(f'Lag Histogram ({ii} frames after flash)')

        hist_list = df_lag.apply(RCtools.compute_cds_histogram, axis=1, roi=roi, gainfits=gainfits, bins=bins).tolist()

        for (y, v) in zip(hist_list, df_lag[TGKEY]):
            plt.plot(x, y, label=v)

        plt.legend(title=f'{TGKEY}', loc='upper right')
        plt.xlabel(f'Lag ({units})')
        plt.ylabel('Frequency')

        # plt.show()
        plt.savefig(outpng)
        print('Saved '+outpng)


if __name__ == '__main__':
    main()
