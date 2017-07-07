from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astrometry.util.fits import fits_table, merge_tables
import os
import numpy as np

base=os.getenv('LEGACY_SURVEY_DIR')

for camera in ['bok']: #,'mosaic','decam']:
    if camera == 'bok':
        fn= os.path.join(base,'survey-ccds-dr4-90prime.fits.gz')
        bands=['g','r']
        zp0= dict(g = 25.74,r = 25.52)
        delta= [0.5,0.18]
        #xlim=[24.8,26]
        #final=['auto',25.9]
    elif camera == 'mosaic':
        fn= os.path.join(base,'survey-ccds-dr4-mzlsv2.fits.gz')
        bands=['z']
        zp0= dict(z= 26.20)
        delta= [0.6,0.6]
        #xlim=[25.4,27]
        #final=['auto',26.8]
    elif camera == 'decam':
        fn= os.path.join(base,'survey-ccds-decals-extra-nondecals.fits.gz')
        bands=['g','r','z']
        zp0= dict(g = 25.08,r = 25.29,z = 24.92)
        delta= [0.5,0.25]
    ccds= fits_table(fn)
    fullsz= len(ccds)
    # Obvious cuts
    keep= (ccds.exptime >= 30)*\
          (ccds.ccdnmatch >= 20)*\
          (np.abs(ccds.zpt - ccds.ccdzpt) <= 0.1)
    perc= 100*np.where(keep == False)[0].size/float(fullsz)
    print('%f percent of full ccd removed with obvious cuts' % perc)
    ccds.cut(keep)
    for band in bands:
        final= [ zp0[band]-delta[0],zp0[band]+delta[1] ]
        xlim=[ final[0]-0.2,final[1]+0.2 ]
        cut= ccds.filter == band
        # Histogram
        bins=np.linspace(xlim[0],xlim[1],num=100)
        fig,ax= plt.subplots()
        ax.hist(ccds.ccdzpt[cut],bins=bins)
        # Median and 90% of everything
        med=np.percentile(ccds.ccdzpt[cut],q=50)
        bot5=np.percentile(ccds.ccdzpt[cut],q=3.5)
        top5=np.percentile(ccds.ccdzpt[cut],q=96.5)
        #ax.plot([med]*2,ax.get_ylim(),'r--',lw=3)
        ax.plot([bot5]*2,ax.get_ylim(),'r--',lw=3)
        ax.plot([top5]*2,ax.get_ylim(),'r--',lw=3)
        print('bot=%f, med=%f, top=%f' % (bot5,med,top5))
        # Draw final cuts
        if final[0] == 'auto':
            final[0]= bot5
        ax.plot([zp0[band]]*2,ax.get_ylim(),'g--',lw=3)
        ax.plot([final[0]]*2,ax.get_ylim(),'m--',lw=3)
        ax.plot([final[1]]*2,ax.get_ylim(),'m--',lw=3)
        keep= (ccds.ccdzpt[cut] >= final[0])*\
              (ccds.ccdzpt[cut] <= final[1])
        perc= 100*np.where(keep == False)[0].size/float(len(ccds[cut]))
        print('%f percent removed in final cut' % perc)
        # save
        ax.set_xlim(xlim)
        name='ccdzpt_%s_%s.png' % (camera,band)
        plt.savefig(name)
        plt.close()
        print('Wrote %s' % name)
print('done')
