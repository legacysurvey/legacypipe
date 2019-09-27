'''
A script for fixing some header & data model errors in DR8 forced photometry results.

# EXTNAME = 'FORCED_PHOTOM'
# units on RA,DEC
# uniformize CAMERA, CCDNAME, and EXPNUM

'''

from astrometry.util.fits import *
from astrometry.util.file import trymakedirs
from glob import glob
import os
import fitsio
import numpy as np

indir = '/global/cscratch1/sd/dstn/dr8-forced/forced'
outdir = '/global/cscratch1/sd/dstn/dr8-forced-fixed'

#fns = glob(indir + '/{90prime,decam,mosaic}/*/*.fits')

#fns = glob(indir + '/*/*/*.fits')
#fns = glob(indir + '/decam/00500/*.fits')
fns = glob(indir + '/mosaic/00064/*.fits')

#fns = glob(indir + '/decam/00500/*.fits')
#print(len(fns))
fns.sort()

for ifn,fn in enumerate(fns):
    print('Reading', ifn+1, 'of', len(fns), ':', fn)
    
    outfn = fn.replace(indir, outdir)
    dirname = os.path.dirname(outfn)
    trymakedirs(dirname)
    T = fits_table(fn)
    hdr = T.get_header()
    primhdr = fitsio.read_header(fn)
    #print('Header:', hdr)

    T.camera = T.camera.astype('S7')  # '90prime'
    T.ccdname = T.ccdname.astype('S4')   # 'CCD4'
    T.expnum = T.expnum.astype(np.int64)
    T.ccd_cuts = T.ccd_cuts.astype(np.int64)

    units = dict()
    for i,c in enumerate(T.get_columns()):
        u = hdr.get('TUNIT%i' % (i+1))
        if u is None:
            continue
        units[c] = u
    units.update(ra='deg', dec='deg')

    # dict -> list
    units = [units.get(c,'') for c in T.get_columns()]

    T.writeto(outfn, primheader=primhdr, header=hdr, extname='FORCED_PHOTOM', units=units)

    print('Wrote  ', ifn+1, 'of', len(fns), ':', outfn)
    
