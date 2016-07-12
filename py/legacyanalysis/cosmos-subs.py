from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from astrometry.util.fits import fits_table, merge_tables
import numpy as np
import pylab as plt

from legacypipe.common import *
from legacypipe.decam import DecamImage

from tractor.sfd import SFDMap

'''
This script selects subsets of images covering the COSMOS region, so
that we can explore how the pipeline performs on repeated imaging of
the same region.  This script computes how much per-pixel noise should
be added so that we just reach our depth targets.  The result is a
FITS table that can be used as the "CCDs" table for the runcosmos.py
script.
'''

bands = 'grz'

# Target depths (90th percentile = 3-pass coverage), for 5-sigma galaxy profile.
target = dict(g=24.0, r=23.4, z=22.5)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--region', default='cosmos', help='Region: "cosmos", "s82qso"')
opt = parser.parse_args()

region = opt.region

# We cache the table of good CCDs in the COSMOS region in this file...
cfn = '%s-all-ccds.fits' % region
if not os.path.exists(cfn):
    #survey = LegacySurveyData(version='dr2')
    survey = LegacySurveyData()
    
    C = survey.get_annotated_ccds()
    print(len(C), 'annotated CCDs')

    if region == 'cosmos':
        C.cut(np.hypot(C.ra_bore - 150, C.dec_bore - 2.2) < 1.)
    elif region == 's82qso':
        #C.cut((C.ra > 35.75) * (C.ra < 42.25) * (C.dec > -1.5) * (C.dec < 1.5))
        C.cut((C.ra > 36) * (C.ra < 36.5) * (C.dec > 0) * (C.dec < 0.5))
    else:
        assert(False)

    print(len(C), 'CCDs in', region)

    C.cut(np.array([f in bands for f in C.filter]))
    print(len(C), 'in', bands)

    C.cut(C.exptime >= 50.)
    print(len(C), 'with exptime >= 50 sec')

    C.cut(survey.photometric_ccds(C))
    print(len(C), 'photometric')

    C.cut(np.lexsort((C.expnum, C.filter)))
    C.writeto(cfn)
    
else:
    C = fits_table(cfn)

plt.clf()
plt.plot(C.ra, C.dec, 'bo', alpha=0.2)
plt.savefig('rd.png')

# Find the unique exposures (not CCDs), save as E.
C.galnorm = C.galnorm_mean
nil,I = np.unique(C.expnum, return_index=True)
E = C[I]
print(len(E), 'exposures')

if False:
    # Find the extinction in the center of the COSMOS region and apply it
    # as a correction to our target depths (so that we reach that depth
    # for de-reddened mags).
    print('Reading SFD maps...')
    sfd = SFDMap()
    filts = ['%s %s' % ('DES', f) for f in bands]
    ext = sfd.extinction(filts, 150., 2.2)
    print('Extinction:', ext)
    # -> Extinction: [[ 0.06293296  0.04239261  0.02371245]]

E.index = np.arange(len(E))
E.passnum = np.zeros(len(E), np.uint8)
E.depthfraction = np.zeros(len(E), np.float32)

# Compute which pass number each exposure would be called.
zp0 = DecamImage.nominal_zeropoints()
# HACK -- copied from obsbot
kx = dict(g = 0.178, r = 0.094, z = 0.060,)
for band in bands:
    B = E[E.filter == band]
    B.cut(np.argsort(B.seeing))

    Nsigma = 5.
    sig = NanoMaggies.magToNanomaggies(target[band]) / Nsigma
    targetiv = 1./sig**2
    
    for exp in B:
        thisdetiv = 1. / (exp.sig1 / exp.galnorm)**2
        # Which pass number would this image be assigned?
        trans = 10.**(-0.4 * (zp0[band] - exp.ccdzpt
                              - kx[band]*(exp.airmass - 1.)))
        seeing_good = exp.seeing < 1.3
        seeing_fair = exp.seeing < 2.0
        trans_good = trans > 0.9
        trans_fair = trans > 0.7
        if seeing_good and trans_good:
            E.passnum[exp.index] = 1
        elif ((seeing_good and trans_fair) or (seeing_fair and trans_good)):
            E.passnum[exp.index] = 2
        else:
            E.passnum[exp.index] = 3
        E.depthfraction[exp.index] = (thisdetiv / targetiv)

for band in bands:
    B = E[E.filter == band]
    B.cut(np.lexsort((B.seeing, (B.depthfraction < 0.34), B.passnum)))
    print(len(B), 'exposures in', band, 'band')
    for exp in B:
        print('  e', exp.expnum, 'see %.2f' % exp.seeing,
              't %3.0f' % exp.exptime, 'pid', exp.propid,
              'object', exp.object,
              'f.depth: %.2f' % exp.depthfraction, #'sig1 %.4f' % exp.sig1,
              'pass', exp.passnum, ('X' if exp.depthfraction < 0.34 else ''))


###
#
#  A second set of 3 specially tailored sets of exposures --
#  with a mix of approximately one image from passes 1,2,3
#  And no overlap in exposures from the first set of 0-4.
#
#  These are called subsets 30, 31, 32.
#
#  We also create no-added-noise ones with the same exposures,
#  called subsets 40, 41, 42.
#
#  The 30/31/32 subset differs very slightly (substituted two
#  exposures) compared to 10/11/12.
###
subset_offset = 30

exposures = [397525, 397526, 511250, # g,p1
             283978, 283979, 283982, # g,p2   -- 283979 was 431103
             289050, 289196, 289155, # g,p3
             405290, 397524, 405291, # r,p1
             397551, 397522, 397552, # r,p2
             397523, 405262, 405292, # r,p3
             180583, 405257, 180582, # z,p1
             180585, 395347, 405254, # z,p2
             #179975, 179971, 179972, # z,p3 -- THESE ONES HAVE NASTY SKY GRADIENTS
             193204, 193205, 192768, # z, p3 -- 193205 was 193180
             ]
# reorder to get one of p1,p2,p3 in each set.  (In the code below we
# keep adding the next exposure until we reach the desired depth --
# here we're setting up the exposure list so that it selects the ones
# we want.)
exposures = exposures[::3] + exposures[1::3] + exposures[2::3]

# Pull out our exposures into table E.
I = []
for e in exposures:
    print('expnum', e)
    I.append(np.flatnonzero(E.expnum == e)[0])
I = np.array(I)
E = E[I]
print('Cut to', len(E), 'exposures')

# Here's the main code that builds the subsets of exposures.
sets = []
for iset in xrange(100):
    print('-------------------------------------------')

    thisset = []
    used_expnums = []
    
    for band in bands:
        gotband = False

        B = E[E.filter == band]
        print(len(B), 'exposures in', band, 'band')

        Nsigma = 5.
        sig = NanoMaggies.magToNanomaggies(target[band]) / Nsigma
        targetiv = 1./sig**2

        maxfrac = 0.34
        detiv = 0.
        for i in range(len(B)):
            exp = B[i]
            print('Image', exp.expnum, 'propid', exp.propid, 'exptime',
                  exp.exptime, 'seeing', exp.seeing)
            # Grab the individual CCDs in this exposure.
            ccds = C[C.expnum == exp.expnum]
            print(len(ccds), 'CCDs in this exposure')

            thisdetiv = 1. / (ccds.sig1 / ccds.galnorm)**2
            print('  mean detiv', np.mean(thisdetiv),
                  '= fraction of target: %.2f' % (np.mean(thisdetiv)/ targetiv))
            
            maxiv = maxfrac * targetiv

            # Compute how much noise should be added so that these CCDs hit
            # the desired depth.
            targetsig1 = ccds.sig1 * np.sqrt(np.maximum(1, thisdetiv/maxiv))
            ccds.addnoise = np.sqrt(np.maximum(0, targetsig1**2 - ccds.sig1**2))
            thisdetiv = 1./(np.hypot(ccds.sig1, ccds.addnoise) /ccds.galnorm)**2
            print('  adding factor %.2f more noise' %
                  (np.mean(ccds.addnoise / ccds.sig1)))
            print('  final detiv: range %g, %g' %
                  (thisdetiv.min(), thisdetiv.max()))
            print('  detivs:', sorted(thisdetiv))
            detiv += np.mean(thisdetiv)
            thisset.append(ccds)
            used_expnums.append(exp.expnum)
            if detiv > targetiv:
                gotband = True
                break
        if not gotband:
            break
    if not gotband:
        break

    Cset = merge_tables(thisset)
    Cset.camera = np.array([c + '+noise' for c in Cset.camera])
    sets.append(Cset)
    
    E.cut(np.array([expnum not in used_expnums for expnum in E.expnum]))
    print('Cut to', len(E), 'remaining exposures')


print('Got', len(sets), 'sets of exposures')

for i,C in enumerate(sets):
    C.writeto('%s-ccds-sub%i.fits' % (region, i))
    C.subset = np.array([subset_offset + i] * len(C)).astype(np.uint8)
    
C = merge_tables(sets)
C.writeto('%s-ccds.fits' % region)

# Add a copy of this subset without adding noise
C2 = C.copy()
C2.subset += 10
C2.addnoise[:] = 0.
C = merge_tables([C, C2])
C.writeto('%s-ccds-2.fits' % region)

#for i,E in enumerate(sets):
#    E.writeto('cosmos-subset-%i.fits' % i)

