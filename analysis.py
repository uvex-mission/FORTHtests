import os

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
from matplotlib import rcParams
from matplotlib.patches import Rectangle
rcParams.update({'font.size': 11})
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from scipy import stats

def dict_subset(d, pattern):
    return {k:d[k] for k in d if pattern in k}

def image_stats(img, sigclip=None, units='ADU', thresh={}):
    '''Generate basic stats for any dimension array'''
    summary = {}
    summary['units']=units

    if sigclip:
        img = stats.sigmaclip(img, sigclip, sigclip)[0]

    # Convert to float(64) to ensure JSON compatibility (doesn't like float32)
    summary['count']     = int(np.size(img))
    summary['mean']      = float(np.mean(img))
    summary['median']    = float(np.median(img))
    summary['std']       = float(np.std(img))
    summary['stderr']    = float((np.std(img)/np.sqrt(np.size(img))))
    summary['sigclip']   = sigclip

    percentiles = [90, 95, 99, 99.9]

    # Create a key and value for each percentile, e.g. '99% : 123'
    summary['percentiles'] = { f'{pp}%' : np.percentile(img, pp) for pp in percentiles }

    # Compute fraction of pixels above each threshhold
    for k,v in thresh.items():
        summary[k] = (img>v).sum() / img.size

    return summary

def print_dict_numerical(mydict, keywidth=10, sigfig=4):
    '''Nicely print the numerical items in a dictionary'''

    msg = ''

    if 'units' in mydict:
        msg += f"{'units'.ljust(keywidth)} = {mydict['units']}\n"

    for k,v in mydict.items():
    	# if isinstance(v, (int, float)):
        try:
            _ = float(v)
    		# print(f"{k.ljust(keywidth)} = {v:.4g}")
            msg += f"{k.ljust(keywidth)} = {v:.4g}\n"
        except:
            pass

    # Recursively print sub-dicts
    for k,v in mydict.items():
        if isinstance(v, dict):
            msg += '\n'+k+'\n'
            msg += print_dict_numerical(v)

    return msg

def makeImageHistogram(img ,img_stats=None ,nbins=100, title=None, units='ADU', log_extend=2, style='loglinear', compact=False):
    '''Histogram with log scale in X and Y -- higher X bins are also wider
    Optional sigma-clipping only used to print summary and estimate plot scale

    Parameters
    ----------
    img : array_like or CCDData
        histogram input data
    img_stats : dict
        If you've already calculated summary stats, input it here to avoid recalculating
    nbins : int
        Number of histogram bins
    units : str
        Units for pixel values; used on x-axis label
    log_extend : float
        Number of decades to extend x-axis past 99.9% quantile
    style : string
        Which of the 2 plot types (log, linear) to include. 'log', 'linear', 'loglinear'
    compact : bool
        If true, keep the x-range small (ignore outliers) on log plot

    Returns
    -------
    fig : figure
        pyplot figure
    '''
    flatimg = img.flatten()

    #Get some stats to help determine plot X ranges
    if img_stats is None:
        img_stats = image_stats(img)

    std = image_stats(img, sigclip=3)['std']  # rough width of the main peak; this step is a bit wasteful

    # Log bins; bin the full range but limit plot range
    # histmax = np.log10(  max(abs(img).max(), std)  )
    histmax = np.log10(img_stats['percentiles']['99.9%']) + log_extend
    histmax = min(histmax, np.log10(abs(img).max()))  # No point in extending past the actual max
    histmin = np.log10(  max(abs(img).min(), std/10)  )
    dbin =  (histmax-histmin)/nbins
    bins = 10**np.arange(histmin ,histmax ,dbin) # Sensible range for display
    log_range = [bins.min(), bins.max()]

    # Add the extremes for bookkeeping
    if img.min() < bins.min():
        bins = np.concatenate(  ([img.min()], bins)  )
    if img.max() > bins.max():
        bins = np.concatenate(  (bins, [img.max()])  )
    log_range[-1] *= 1.5  # Adjust plot range to show pileup on right side

    if compact:
        log_range[-1] = img_stats['percentiles']['99%']

    # Which plot types to use
    DOLOG = 'log' in style.lower()
    DOLINEAR = 'linear' in style.lower()
    figscale = 4

    fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[3, 2], figsize=(3*figscale,1*figscale))

    #Log histogram
    if DOLOG:

        # Plot + and - histograms; values < 0 are skipped by hist()
        p_wide=ax1.hist(flatimg ,bins=bins ,log=True ,color='lightblue' ,alpha=1 ,label='+')
        p_wide_neg=ax1.hist(-flatimg ,bins=bins ,log=True ,color='grey' ,alpha=.5 ,label='—')

        ax1.set_xscale('log')
        ax1.set_xlim(log_range)
        ax1.set_xlabel(units)
        ax1.set_ylabel('# pixels')
        ax1.legend(loc='upper left' ) #,bbox_to_anchor=(0,1)

        # Cumulative plot
        ax1c = ax1.twinx()
        p_cumu=ax1c.hist(flatimg ,bins=bins ,color='darkred', cumulative=True, density=True, histtype='step')

    else:
        fig.delaxes(ax1)

    #Linear bins; bin the full range but limit display
    if DOLINEAR:

        # zoom_range = img_stats['median'] + np.array([-1,1])*std*3
        dhigh = img_stats['percentiles']['99.9%'] - img_stats['median']
        zoom_range = [img_stats['median']-std*3, img_stats['median']+1.5*dhigh] # Sensible range for display
        zoom_range[0] = max(zoom_range[0], img.min())
        zoom_range[1] = min(zoom_range[1], img.max()) # No point in extending past the actual max/min

        zoom_bins = np.arange(min(zoom_range), max(zoom_range),std/4 )

        zoom_range[-1]*=1.05  # Adjust plot range to show pileup on right side

        # Add the extremes for bookkeeping
        if img.min() < zoom_bins.min():
            zoom_bins = np.concatenate(  ([img.min()], zoom_bins)  )
        if img.max() > zoom_bins.max():
            zoom_bins = np.concatenate(  (zoom_bins, [img.max()])  )

        # Tail on Linear scale won't be visible past a certain point
        zoom_range[-1] = img_stats['percentiles']['99%']
            
        # Linear scale
        p_zoom=ax2.hist(flatimg ,bins=zoom_bins ,color='lightblue')
        ax2.set_xlabel(units)
        ax2.set_xlim(zoom_range)

        # Cumulative plot
        ax2c = ax2.twinx()
        p_zoom=ax2c.hist(flatimg ,bins=zoom_bins ,color='darkred', cumulative=True, density=True, histtype='step')

        # Remove y-axis between the linear and log plots
        ax2.set_yticks([])
        ax2.set_ylabel('Relative')

        #Mark the median
        x=img_stats['median']
        ax2.axvline(x=x, color="black" ,linestyle='--')
        sigma_text = f'  Median %.4g %s'%(x ,units)
        # dum = ax2.text(x*.95 ,p_zoom[0].max() , sigma_text ,verticalalignment='top', horizontalalignment='right' ,fontsize=12)
        ax2.set_title(sigma_text, fontsize=12)
    else:
        fig.delaxes(ax2)

    fig.suptitle(title, y=1.05)
    fig.tight_layout()  #pad=5.0

    return fig

def minus_plus(x, y): return x-y, x+y

def context_image(image_array, img_stats=None, figscale=15, sig=3, units='ADU', title=None, **kwargs):
    ''' Nicely display a detector image 

    Parameters
    ----------
    img_array : array_like
        2D image
    img_stats : dict
        Pre-calculated summary stats as a dictionary
    figscale  : float
        Image size of the largest axis
    sig       : float
        Number or sigmas to set color scale limits
    units     : String
        Colorbar units label
    title     : String
        Plot title
    **kwargs  : Extra keywords for imshow()
    '''

    kwargs_default = {
        'interpolation' : 'none',
        'cmap' : 'viridis',
        'origin' : 'lower'
    }

    figshape = np.array(np.array(image_array).shape)
    figsize = (figshape.T * figscale/max(figshape)).tolist()  # Scale so largest axis = figscale
    # figratio = figsize[0]/figsize[1]
    
    fig = plt.figure(figsize=figsize)  # figsize is (width, height) unlike 2D python arrays

    if img_stats:
        vmin, vmax = minus_plus(img_stats['median'], sig*img_stats['std'])

        # No need to extend scale past the actual range of the image
        kwargs_default['vmin'] = max(vmin, image_array.min())
        kwargs_default['vmax'] = min(vmax, image_array.max())

    # Merge keyword dicts
    for k,v in kwargs_default.items():
        if k not in kwargs:
            kwargs[k] = v

    # plt.imshow(image_array, cmap=cmap, origin='lower', interpolation='none', vmin=vmin, vmax=vmax, **kwargs)
    plt.imshow(image_array, **kwargs)
    # plt.colorbar(label=units, fraction=figratio*0.047)  # Magic ratio for matching the colorbar to the image

    # plt.title(title)
    plt.xticks([0,figshape[1]], [0,figshape[1]])
    plt.yticks([0,figshape[0]], [0,figshape[0]])
    plt.yticks(rotation=90)
    plt.xlabel(units)  # Use x-axis as colorbar label

    # add color bar below chart
    divider = make_axes_locatable(fig.axes[0])
    cax = divider.new_vertical(size='5%', pad=0.5, pack_start = True)
    fig.add_axes(cax)
    plt.colorbar(cax = cax, orientation = 'horizontal')

    return fig, None

def saveplot(fig, fname, outdir, printlog=print, close=True):
    ''' Helper function for saving pyplot figures 

    Parameters
    ----------
    fig : pyplot figure object

    outdir : string
        Path to output directory
    fname : string
        Output file basename
    title : string
        Title displayed at top
    printlog : function
        Use this function to print and log the result.  Default is python print()
    close : bool
        Close the pyplot figure (passed by reference to this function) to free up memory
    '''
    outfile = os.path.join(outdir, fname)

    fig.savefig(outfile,
        # dpi=300,              # High resolution
        # format='png',         # Explicit format
        bbox_inches='tight',  # Tight layout
        pad_inches=0.1,       # Padding
        facecolor='white',    # Background color
        edgecolor='none')     # Edge color

    printlog(outfile)

    if close: plt.close(fig)


def image_summary(img, fname_img, fname_hist, outdir, 
                    units='ADU', thresh={}, printlog=print, title='', sigclip=3):
    ''' Generate context image, histogram, and stats tables for a 2D array image

    img     : 2D array
    fname_img   : Filename string (no path) for full frame
    fname_hist  : Filename string (no path) for histogram
    outdir      : Path string to output files
    '''

    # Calculate and print basic statistics
    stats_full = image_stats(img, units=units, thresh=thresh)
    printlog('\nFull Stats:')
    printlog(print_dict_numerical(stats_full))

    stats_clip = image_stats(img, units=units, thresh=thresh, sigclip=sigclip)
    printlog(f'Clipped ({sigclip}) Stats:')
    printlog(print_dict_numerical(stats_clip))

    # Make context image
    fig, ax = context_image(img, units=units, img_stats=stats_clip, title=title)
    saveplot(fig, fname_img, outdir, printlog)

    # Make histogram
    fig = makeImageHistogram(img, units=units, img_stats=stats_clip, title=title)
    saveplot(fig, fname_hist, outdir, printlog)

    # Format for report
    ret = {
        'stats_full'    :   stats_full,
        'stats_clip'    :   stats_clip,
        'fname_img'     :   fname_img,
        'fname_hist'    :   fname_hist
    }
    return ret

