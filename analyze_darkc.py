import argparse
import os
import sys

import astropy.io.fits as pf
import matplotlib.pyplot as plt
import numpy as np

import RCtools

timekey = 'EXPTIME'
SIGMACLIP = 3.
NSKIP = 2

roi_full = slice(None)

##########################################

def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze DARKC FITS file'
    )
    parser.add_argument(
        'files',
        nargs='+',
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
    return parser.parse_args()


def main():
    args = parse_args()

    xmin, xmax, ymin, ymax = args.roi
    roi_arg = (slice(xmin,xmax),slice(ymin,ymax))
    roilist = [roi_full, roi_arg]

    df = RCtools.all_headers_to_df(args.files)
    header_list = ['FILENAME', 'EXTN', 'EXTNAME', timekey, 'GAINFITS']
    df = df[header_list]

    # 1st 2 images may be unreliable.  Next images are short darks i.e. long biases
    # Final image is the long dark
    biascube = []
    with pf.open(args.files) as hdulist:

        imgdark = hdulist[-1]
        imgdark = imgdark[1] - imgdark[0]

        for hdu in hdulist[1+NSKIP:-1]:
            img = hdu.data
            img = img[1] - img[0]
            biascube.append(img)
            del img

    biascube = np.array(biascube)
    imgbias = np.median(imgbias, axis=0)
    del biascube

    # Bias subtraction
    imgdark = imgdark - imgbias

    for roi in roilist:

        img = imgdark[roi]

        # Compute statistics, histograms



    # ── Save CSV ─────────────────────────────────────────────────────────
    outcsv = stem + '.csv'
    df.to_csv(outcsv, index=False)
    print(f'Saved: {outcsv}')

    # ── Plot Lag vs. stimulus for each VHIGH_TG ──────────────────────────────
    outpng = stem + '.png'

    plt.figure(figsize=(5, 10))

    plt.xscale('log')
    plt.xlabel('Stimulus (ADU)')
    plt.ylabel('Lag_0 (ADU)')
    plt.legend()
    plt.savefig(outpng)
    plt.show()
    print(f'Saved: {outpng}')


if __name__ == '__main__':
    main()
