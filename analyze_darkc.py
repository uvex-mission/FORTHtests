import argparse
import os
import sys

import astropy.io.fits as pf
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import RCtools
import analysis as cca  # catcam analysis

TIMEKEY = 'EXPTIME'
GAINKEY = 'GAINFITS'
SIGMACLIP = 3.
NSKIP = 2

roi_full = slice(None)
roilist = [roi_full]

OUTNAME = 'darkc'

##########################################

def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze DARKC FITS file'
    )
    parser.add_argument(
        'files',
        # nargs=1,
        metavar='FILE',
        help=(
            'DARKC FITS file'
        )
    )
    parser.add_argument(
        '--roi',
        nargs=4,
        type=int,
        default=None,
        metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
        help=(
            'Required analysis region in pixel coordinates: XMIN XMAX YMIN YMAX (column and row bounds). '
        )
    )
    parser.add_argument(
        '--outdir',
        default='./',
        metavar='DIR',
        help='Output directory for CSV and PNG files. Defaults to ./'
    )
    parser.add_argument(
        '--compact',
        action='store_true',
        help='Include histograms that ignore outliers'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.roi:
        xmin, xmax, ymin, ymax = args.roi
        roi_arg = (slice(ymin,ymax),slice(xmin,xmax))
        roilist.append(roi_arg)

    df = RCtools.all_headers_to_df(args.files)
    header_list = ['FILENAME', 'EXTN', 'EXTNAME', TIMEKEY, 'GAINFITS']
    df = df[header_list]

    # 1st 2 images may be unreliable.  Next images are short darks i.e. long biases
    # Final image is the long dark
    biascube = []
    print(args.files)
    with pf.open(args.files) as hdulist:

        darkhdu = hdulist[-1]
        imgdark = darkhdu.data.astype('float32')
        imgdark = imgdark[1] - imgdark[0]

        for hdu in hdulist[1+NSKIP:-1]:
            img = hdu.data.astype('float32')
            img = img[1] - img[0]
            biascube.append(img)
            del img

    biascube = np.array(biascube)
    imgbias = np.median(biascube, axis=0)
    del biascube

    t_dark = darkhdu.header[TIMEKEY]
    t_bias = hdu.header[TIMEKEY]
    dt = t_dark - t_bias

    # Bias subtraction, time scaling, gain 
    imgdark = (imgdark - imgbias)/dt * darkhdu.header[GAINKEY]  # e-/px/s
    units = 'e-/s'

    pf.writeto(OUTNAME+'.fits', imgdark, header=darkhdu.header, overwrite=True)
    print('Saved', OUTNAME+'.fits')

    img_stats = cca.image_stats(imgdark, sigclip=3, units=units)
    fig, _ = cca.context_image(imgdark, img_stats=img_stats, figscale=15, sig=3, units=units, title=None)

    # Overlay ROIs (skip roilist[0], the full-frame ROI)
    ax = fig.axes[0]
    for i, roi in enumerate(roilist):
        if i == 0:
            continue
        x0, y0 = roi[1].start, roi[0].start
        width = roi[1].stop - roi[1].start
        height = roi[0].stop - roi[0].start
        rect = patches.Rectangle((x0, y0), width, height,
                                 linewidth=1.5, linestyle='--',
                                 edgecolor='darkred', facecolor='none')
        ax.add_patch(rect)
        ax.text(x0 + 0.02 * width, y0 + height - 0.02 * height, f'ROI {i}',
                color='darkred', va='top', ha='left', fontsize=14)

    cca.saveplot(fig, OUTNAME+'.png', args.outdir)

    rows = []

    for i, roi in enumerate(roilist):

        img = imgdark[roi]

        # Compute statistics, histograms
        stats = RCtools.compute_stats(img, clip_sigma=0, do_median=True, quantiles=[90,99,99.9])
        rows.append(stats)

        outpng = f'{OUTNAME}_hist_roi{i}.png'

        if i == 0:
            roi_title = 'ROI 0 (full frame)'
        else:
            rows_sl, cols_sl = roi
            roi_title = (f'ROI {i} (row {rows_sl.start}:{rows_sl.stop}, '
                         f'col {cols_sl.start}:{cols_sl.stop})')

        fig = cca.makeImageHistogram(img ,nbins=100, units=units, compact=False, title=roi_title)
        cca.saveplot(fig, outpng, args.outdir)

        if args.compact:
            fig = cca.makeImageHistogram(img ,nbins=100, units=units, compact=True, title=roi_title)
            cca.saveplot(fig, outpng[:-4]+'_hist_compact.png', args.outdir)

    df_out = pd.DataFrame(rows)
    df_out.index.names = ['ROI']

    # ── Save CSV ─────────────────────────────────────────────────────────
    outcsv = OUTNAME + '.csv'
    df_out.to_csv(outcsv, index=True)
    print(f'Saved: {outcsv}')


if __name__ == '__main__':
    main()
