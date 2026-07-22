'''
R. Cook notes:

DARKTEST – contains a total of 26 readout frames bracketing 25 high-gain-with-TG exposures:
Frames 0-1: bracket a minimum duration exposure aimed at clearing persistence/lag.
Following frames define 8 sets of each consisting of three exposures. 
In each set, the first two exposures are of minimum duration and intended to clear persistence/lag, 
while the third exposure is of duration 30 seconds.
Each of the 8 30 second exposures is performed using a different combination of glow mitigation features, 
testing all combinations of three mitigations:
	1. Turning off VHIGH_BIAS (VHIGH_ROW, VHIGH_MIM and VHIGH_RST)
	2. Turning off all the pixel source follower current sources
	3. Turning off all the Y_ENABLE_BLOCK signals.
A fourth feature (turning off the detector output PMOS source follower current sources) is always used, 
but is known to have no impact on glow.
The optimal combo of mitigations always include #2 above, leaving only four options to choose from.
Analysis could just consist of selecting the best combo via visual inspection.

NOGLO modes;  ints 0-15; 0-7 all bad ### 14 is best (TBC)
8: off
9: Vhighs off
10: I_PIX_PAD=3.3V
11: Vhighs off + I_PIX_PAD=3.3V
12-15: Same order as above + YBLK_EN’s off
'''

import sys

if len(sys.argv)<2:
	print(f'USAGE:  python {__file__} bigDarktestFile.fits')
	sys.exit()
 
VMAX = sys.argv[2] if len(sys.argv)>2 else None

import astropy.io.fits as pf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import RCtools

fname = sys.argv[1]
OUTFILE = 'darktest.png'

timekey = 'EXPTIME'
t_short = 6. # [s] Approx. length of the short exposures we want to skip
t_long = 30. # [s] Approx. length of the exposures we want to analyze
PERCENTILE = 95  # Max scale for displayed images
UNITS = 'ADU ~e'
N_NOGLO = 8  # Expected number of images to check

#---------------------------------------------------

# Organize header data
df = RCtools.all_headers_to_df(fname)
t_mid = (t_short+t_long)/2
df_img = df[df[timekey]>t_mid] # somewhat robust way of selecting the longer exposures

assert len(df_img) == N_NOGLO, f"Exactly {N_NOGLO} images required."

print('Loading data...')
imgcube = np.array([pf.getdata(fname, extn).astype(int) for extn in df_img['EXTN'] ])
imgcube = np.swapaxes(imgcube, 0, 1)
imgcube = imgcube[1]-imgcube[0]  # CDS image cube

titles = [f'NOGLO {i+8}' for i in range(len(imgcube))]

percentX = np.array([ np.percentile(img, 95) for img in imgcube ])
vmin=0
vmax = VMAX if VMAX else sorted(percentX)[-2]  # 2nd highest value -- we know highest is an outlier


def plot_array_grid(arrays, titles=titles, figsize=(16, 8), **kwargs):
    """
    Display a list of 8 2D arrays in a 4-column × 2-row grid,
    each with a colorbar matched to its image width.

    Parameters
    ----------
    arrays : list of 8 2D np.ndarray
    titles  : optional list of 8 strings
    figsize : figure size tuple
    kwargs:  extra keywords for imshow()
    """
    assert len(arrays) == N_NOGLO, f"Exactly {N_NOGLO} images required."

    fig, axes = plt.subplots(2, 4, figsize=figsize)

    for i, (ax, arr) in enumerate(zip(axes.flat, arrays)):
        im = ax.imshow(arr, origin='lower', **kwargs)

        if titles:
            ax.set_title(titles[i])

        # Remove tick labels but keep ticks
        ax.tick_params(axis='x', labelbottom=False)

        # Attach a colorbar that exactly matches the image width
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="horizontal", label=UNITS)

    fig.tight_layout()
    return fig

print('Plotting...')
fig = plot_array_grid(imgcube, titles=titles, vmin=vmin, vmax=vmax)

plt.savefig(OUTFILE)
print('SAVED:',OUTFILE)

plt.show()  # This will pop-up a display window
