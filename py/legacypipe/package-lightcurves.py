from __future__ import print_function
import os
import numpy as np
from legacypipe.survey import *
from astrometry.util.util import Tan
from astrometry.util.fits import *


forced_dir = 'forced-dr3'

ra,dec = 29.9902, 0.5529

dra = ddec = 0.001

ralo,  rahi  = ra  - dra , ra  + dra
declo, dechi = dec - ddec, dec + ddec

survey = LegacySurveyData()

bricks = survey.get_bricks_readonly()
I = np.flatnonzero((bricks.ra1  <= rahi ) * (bricks.ra2  >= ralo) *
                   (bricks.dec1 <= dechi) * (bricks.dec2 >= declo))
print(len(I), 'bricks overlap ROI')

TT = []
for b in bricks[I]:
    fn = survey.find_file('tractor', brick=b.brickname)
    print('Reading', fn)
    T = fits_table(fn)
    print('Read', len(T))
    T.cut((T.ra >= ralo) * (T.ra <= rahi) * (T.dec >= declo) * (T.dec <= dechi))
    print(len(T), 'survive cut')
    if len(T) == 0:
        continue
    TT.append(T)
T = TT = merge_tables(TT)

pixscale = dra / 1000.
W,H = int(np.ceil(2.*dra / pixscale)), int(np.ceil(2.*ddec / pixscale))
wcs = Tan(ra, dec, (W+1.)/2., (H+1.)/2., -pixscale, 0., 0., pixscale, float(W), float(H))
print('WCS radec bounds:', wcs.radec_bounds())

idmap = dict([((b,oid),i) for i,(b,oid) in enumerate(zip(T.brickname, T.objid))])
bands = 'grz'
lightcurves = dict([(b, [[] for i in range(len(T))]) for b in bands])

#ccds = survey.get_ccds_readonly()
#I = np.flatnonzero((ccds.ra1  <= rahi ) * (ccds.ra2  >= ralo) *
#                   (ccds.dec1 <= dechi) * (ccds.dec2 >= declo))
ccds = survey.ccds_touching_wcs(wcs)
print(len(ccds), 'ccds overlap ROI')

I = survey.photometric_ccds(ccds)
ccds.cut(I)
print('Cut to', len(ccds), 'CCDs that are photometric')

FF = []
NF = 0
JJ = []
for ccd in ccds:
    print('Expnum', ccd.expnum, 'ccdname', ccd.ccdname)
    print('RA,Dec', ccd.ra, ccd.dec)

    # HACK
    fn = os.path.join(forced_dir, 'forced-%i-%s.fits' % (ccd.expnum, ccd.ccdname))
    if not os.path.exists(fn):
        print('Not found:', fn)
        print('python -u legacypipe/forced_photom_decam.py --apphot %i %s DR3 forced-dr3/forced-%i-%s.fits' % (ccd.expnum, ccd.ccdname, ccd.expnum, ccd.ccdname))
        continue
    print('Reading', fn)
    F = fits_table(fn)
    print('Read', F)

    J = np.array([idmap.get((b,oid), -1) for b,oid in zip(F.brickname, F.objid)])
    I = np.flatnonzero(J >= 0)
    F.cut(I)
    J = J[I]

    F.expnum = np.array([ccd.expnum] * len(F))
    F.ccdname = np.array(['% -3s' % ccd.ccdname] * len(F))

    #print('F.ccdname', F.ccdname)
    
    band = F.filter[0]
    lc = lightcurves[band]

    for i,j in enumerate(J):
        lc[j].append(i + NF)

    FF.append(F)
    NF += len(F)

F = FF = merge_tables(FF)

for band in bands:
    lc = lightcurves[band]
    Nmax = max([len(ii) for ii in lc])
    print('Max number of obs for band', band, ':', Nmax)

    # Set up zeroed arrays
    columns = ['mjd', 'flux', 'flux_ivar', 'expnum', 'ccdname']
    for c in columns:
        #print('Column', c, 'dtype', F.get(c).dtype)
        T.set('lc_%s_%s' % (band,c), np.zeros((len(T), Nmax), F.get(c).dtype))

        X = T.get('lc_%s_%s' % (band, c))
        FX = F.get(c)

        #print('Column:', X)

        for i,jj in enumerate(lc):
            if len(jj) == 0:
                continue
            #X[i,:len(jj)] = FX[np.array(jj)]
            for k,j in enumerate(jj):
                X[i,k] = FX[j]
                #print('Setting element %i,%i to' % (i,k), FX[j])

T.writeto('lc.fits')
