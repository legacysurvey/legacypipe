from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from astrometry.util.fits import fits_table, merge_tables
import numpy as np
import pylab as plt

from legacypipe.survey import *
from legacypipe.decam import DecamImage

from tractor.sfd import SFDMap
from tractor import NanoMaggies

'''
This script selects subsets of images covering the COSMOS region, so
that we can explore how the pipeline performs on repeated imaging of
the same region.  This script computes how much per-pixel noise should
be added so that we just reach our depth targets.  The result is a
FITS table that can be used as the "CCDs" table for the runcosmos.py
script.
'''

bands = 'griz'

# Target depths (90th percentile = 3-pass coverage), for 5-sigma galaxy profile.
target = dict(g=24.0, r=23.4, i=23.0, z=22.5)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--region', default='cosmos', help='Region: "cosmos", "s82qso"')
parser.add_argument('--run', default=None, help='Subset of CCDs files to use: eg "south"')
parser.add_argument('-v', '--verbose', dest='verbose', action='count',
                    default=0, help='Make more verbose')
opt = parser.parse_args()

import sys
import logging
if opt.verbose == 0:
    lvl = logging.INFO
else:
    lvl = logging.DEBUG
logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

region = opt.region

run = opt.run

# We cache the table of good CCDs in the region of interest in this file...
cfn = '%s-all-ccds.fits' % region
if not os.path.exists(cfn):
    from legacypipe.runs import get_survey

    import legacypipe.runs
    print(legacypipe.runs.__file__)
    #survey = LegacySurveyData()
    survey = get_survey(run)

    #C = survey.get_annotated_ccds()
    ## HACK -- fitsio fails to read the .fits.gz file
    # (https://github.com/esheldon/fitsio/issues/255)
    # but can read the uncompressed one.
    print('HACK -- reading ccds-annotated-dr10-v5.fits directly')
    C = fits_table(os.path.join(survey.survey_dir, 'ccds-annotated-dr10-v5.fits'))
    survey.cleanup_ccds_table(C)
    print(len(C), 'annotated CCDs')

    if region == 'cosmos':
        #C.cut(np.hypot(C.ra_bore - 150.1, C.dec_bore - 2.2) < 1.)
        C.cut(np.hypot(C.ra_bore - 150.1, C.dec_bore - 2.2) < 0.3)
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

    #C.cut(survey.photometric_ccds(C))
    #print(len(C), 'photometric')
    #C.cut(C.ccd_cuts == 0)
    C.cut((C.ccd_cuts & ~0x4000) == 0)
    print(len(C), 'pass ccd_cuts==0 or 0x4000')

    C.cut(np.lexsort((C.expnum, C.filter)))
    C.writeto(cfn)
    
else:
    print('Reading cached CCD list from', cfn)
    C = fits_table(cfn)

plt.clf()
plt.plot(C.ra, C.dec, 'bo', alpha=0.2)
plt.savefig('rd.png')

# Drop the "depthcut" bit from ccd_cuts!
C.ccd_cuts = (C.ccd_cuts & ~0x4000)

# Find the unique exposures (not CCDs), save as E.
C.galnorm = C.galnorm_mean
nil,I = np.unique(C.expnum, return_index=True)
E = C[I]
print(len(E), 'exposures')
E.index = np.arange(len(E))
E.passnum = np.zeros(len(E), np.uint8)
E.depthfraction = np.zeros(len(E), np.float32)

E.seeing = E.pixscale_mean * E.fwhm

# Compute which pass number each exposure would be called.
#zp0 = DecamImage.nominal_zeropoints()

# From eyeballing histograms of DR7 zeropoints
#zp0 = dict(g=25.15, r=25.35, z=25.0)
# HACK -- this is copied from obsbot
#kx = dict(g = 0.178, r = 0.094, z = 0.060,)
from legacypipe.decam import DecamImage
zp0 = DecamImage.ZP0.copy()
kx = DecamImage.K_EXT.copy()

for band in bands:
    B = E[E.filter == band]
    B.cut(np.argsort(B.seeing))

    Nsigma = 5.
    sig = NanoMaggies.magToNanomaggies(target[band]) / Nsigma
    targetiv = 1./sig**2

    for exp in B:
        thisdetiv = 1. / (exp.sig1 / exp.galnorm)**2
        # the zps listed in DECamImage are about this much larger than the by-hand
        # values we had in before...
        zp = zp0[band] - 0.15
        # Which pass number would this image be assigned?
        trans = 10.**(-0.4 * (zp - exp.ccdzpt
                              - kx[band]*(max(exp.airmass, 1.) - 1.)))
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
        #print('Band', exp.filter, 'Transparency', trans)

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



if region == 'cosmos':
    # For COSMOS, we hand-select sets of exposures with desired
    # properties.
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
    #
    #
    #    30:
    #    g band:
    #    Image 397525 propid 2014B-0146 exptime 180.0 seeing 1.12011
    #    Image 283978 propid 2014A-0073 exptime 90.0 seeing 1.39842
    #    Image 289050 propid 2014A-0608 exptime 160.0 seeing 1.81853
    #    r band:
    #    Image 405290 propid 2014B-0146 exptime 250.0 seeing 1.23424
    #    Image 397551 propid 2014B-0146 exptime 135.0 seeing 1.27619
    #    Image 397523 propid 2014B-0146 exptime 250.0 seeing 1.30939
    #    z band:
    #    Image 180583 propid 2012B-0003 exptime 300.0 seeing 1.17304
    #    Image 180585 propid 2012B-0003 exptime 300.0 seeing 1.31529
    #    Image 193204 propid 2013A-0741 exptime 300.0 seeing 1.55342
    #    
    #    31:
    #    g band:
    #    Image 397526 propid 2014B-0146 exptime 180.0 seeing 1.15631
    #    Image 283979 propid 2014A-0073 exptime 90.0 seeing 1.42818
    #    Image 289196 propid 2014A-0608 exptime 160.0 seeing 1.82579
    #    r band:
    #    Image 397524 propid 2014B-0146 exptime 250.0 seeing 1.23562
    #    Image 397522 propid 2014B-0146 exptime 250.0 seeing 1.28107
    #    Image 405262 propid 2014B-0146 exptime 250.0 seeing 1.31202
    #    z band:
    #    Image 405257 propid 2014B-0146 exptime 300.0 seeing 1.17682
    #    Image 395347 propid 2014B-0146 exptime 300.0 seeing 1.32482
    #    Image 193205 propid 2013A-0741 exptime 300.0 seeing 1.50034
    #    
    #    32:
    #    g band:
    #    Image 511250 propid 2014B-0404 exptime 89.0 seeing 1.16123
    #    Image 283982 propid 2014A-0073 exptime 90.0 seeing 1.42682
    #    Image 289155 propid 2014A-0608 exptime 160.0 seeing 2.07036
    #    r band:
    #    Image 405291 propid 2014B-0146 exptime 250.0 seeing 1.2382
    #    Image 397552 propid 2014B-0146 exptime 135.0 seeing 1.29461
    #    Image 405292 propid 2014B-0146 exptime 250.0 seeing 1.29947
    #    z band:
    #    Image 180582 propid 2012B-0003 exptime 300.0 seeing 1.17815
    #    Image 405254 propid 2014B-0146 exptime 300.0 seeing 1.33786
    #    Image 192768 propid 2013A-0741 exptime 100.0 seeing 1.57756
    #
    #
    ###
    
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
    
    #subset_offset = 30
    #exposures = exposures[::3] + exposures[1::3] + exposures[2::3]
    
    #
    # Subsets 50, 51, and 52 have, respectively, images from pass 1, pass2, and pass 3.
    #
    #subset_offset = 50
    
    #
    # Subsets 60 through 69 have progressively worse seeing (but still pretty good...)
    #
    #subset_offset = 60

    # DR7
    subset_offset = 70
    
    exposures = [
        # 411355, 411305, 411406, # g, p1 (seeing 1.05-1.1)
        # 397525, 411808, 397526, # g, p1 (seeing 1.1)
        # 511250, 421590, 411456, # g, p1 (seeing 1.2)
        # 397527, 410971, 411707, # g, p1 (seeing 1.2)
        # #411758, 411021, 633993, # g, p1 (seeing 1.2)
        # 
        # 288970, 411055, 410915, # g, p2 (seeing 1.3)
        # 283978, 431103, 283982, # g, p2 (seeing 1.4)
        # 524719, 177358, 524705, # g, p2 (seeing 1.5)
        # 177361, 177085, 177088, # g, p2 (seeing 1.6)
        # 177092, 289278, 177089, # g, p2 (seeing 1.7)
        # 289050, 177091, 289196, # g, p2 (seeing 1.8)

        # DR7 (without depth cut)
        411355, 411406, 411305, # g, p1 (seeing 1.05-1.2)
        397525, 289010, 288970, # g, p1 (seeing 1.2)
        397526, 411808, 411456, # g, p1 (seeing 1.2)
        511250, 397527, 421590, # g, p1 (seeing 1.25)
        411758, 411707, 410971, # g, p1 (seeing 1.3)
        #289444, 411021, 411055, # g, p2 (seeing 1.3)
        413680, 289865, 412604, # g, p2 (seeing 1.4)
        289486, 288929, 289650, # g, p2 (seeing 1.5)
        524719, 524705, 289691, # g, p2 (seeing 1.6)
        524704, 289237, 412554, # g, p2 (seeing 1.7)
        289278, 289907, 289050, # g, p2 (seeing 1.8)

        # 421552, 431105, 405290, # r, p1 (seeing 1.2)
        # 431108, 397524, 405291, # r, p1 (seeing 1.2)
        # 405264, 397553, 405263, # r, p1 (seeing 1.2)
        # 397551, 397522, 397552, # r, p1 (seeing 1.3)
        # 397523, 405262, 431102, # r, p2 (seeing 1.3)
        # 420721, 420722, 177363, # r, p2 (seeing 1.4)
        # 177346, 177362, 524722, # r, p2 (seeing 1.5)
        # 177342, 524707, 177341, # r, p2 (seeing 1.5-1.6)
        # 177343, 524706, 177367, # r, p2 (seeing 1.6-1.7)
        # 413971, 413972, 413973, # r, p2 (seeing 1.8-1.9)
        # r, p3

        # DR7
        421552, 397523, 405290, # r, p1, seeing 1.1
        397524, 397522, 405291, # r, p1, seeing 1.15
        405264, 405263, 397553, # r, p1, seeing 1.2
        397552, 397551, 405292, # r, p1, seeing 1.2
        405262, 431105, 410865, # r, p1, seeing 1.3
        420722, 420721, 524708, # r, p2, seeing 1.4
        431108, 524720, 730075, # r, p2, seeing 1.4
        524721, 524707, 524722, # r, p2, seeing 1.45
        420723, 431102, 524706, # r, p2, seeing 1.5
        413971, 413972, 413973, # r, p2, seeing 2.0

        # 630675, 630928, 431107, # z, p1 (seeing 0.9-1.1)
        # 431104, 431101, 180583, # z, p1 (seeing 1.1)
        # 405257, 180582, 397532, # z, p1 (seeing 1.2)
        # 405256, 397557, 397533, # z, p1 (seeing 1.2)
        # 
        # 524713, 524714, 524716, # z, p2 (seeing 1.26)
        # 180585, 420730, 395347, # z, p2 (seeing 1.32)
        # 413981, 395345, 176837, # z, p2 (seeing 1.4)
        # # 179973, 176845, 193204, # z, p2 (seeing 1.5) (old 67)
        # # 193180, 192768, 453883, # z, p2 (seeing 1.6) (old 68)
        # # 179975, 413978, 453884, # z, p2 (seeing 1.7) (old 69)
        # 176844, 453882, 176845, # z, p2 (seeing 1.5)   (new 67)
        # 193204, 193180, 192768, # z, p2 (seeing 1.6)   (new 68)
        # 453883, 413978, 453884, # z, p2 (seeing 1.7)   (new 69)
        # 
        # # z, p3

        # DR7
        514592, 630675, 514594, # z, p1, seeing 1.0
        514294, 630928, 514591, # z, p1, seeing 1.0
        514889, 514890, 514891, # z, p1, seeing 1.08
        514894, 397535, 420732, # z, p1, seeing 1.2
        397534, 395319, 515786, # z, p1, seeing 1.27
        420730, 431101, 405254, # z, p2, seeing 1.3
        413979, 514584, 395318, # z, p2, seeing 1.38
        395345, 514580, 514581, # z, p2, seeing 1.5
        453882, 413978, 453883, # z, p2, seeing 1.6-1.8
        453884, 525634, 525633, # z, p3, seeing 1.7-2.0
        ]
    # BAD exposures (nasty background gradient)
    # 179971 through 179975


    # DR9
    subset_offset = 80
    exposures = [
        # g, pass 1, seeing ~1.15
        615183, 615180, 615174,
        # g, pass 1, seeing ~1.2
        615177, 615171, 421590,
        # g, pass 1/2, seeing ~1.2
        621609, 614492, 614501,
        # g, pass 2, seeing ~1.2
        614495, 621607, 621973,
        # g, pass 2, seeing ~1.3
        621977, 621605, 621975,
        # g, pass 2, seeing ~1.3
        622361, 622359, 413680,
        # g, pass 2, seeing ~1.3
        622360, 412604, 431106,
        # g, pass 2, seeing ~1.4
        431103, 524717, 524718,
        # g, pass 2, seeing ~1.44
        524703, 524705, 524719,
        # g, pass 2, seeing 1.5
        524704, 412554, 743018,

        # r, pass 1, seeing 1.05
        615172, 615178, 615181,
        # r, pass 1, seeing 1.1
        431105, 615184, 615554,
        # r, pass 1, seeing 1.12
        421552, 405264, 615175,
        # r, pass 1, seeing 1.14
        405263, 405291, 405262,
        # r, pass 1, seeing 1.14
        615553, 397551, 397552,
        # r, pass 1, seeing 1.2
        397524, 410865, 420721,
        # r, pass 2, seeing 1.3
        177362, 177363, 177345,
        # r, pass 2, seeing 1.35
        177344, 177346, 177343,
        # r, pass 2, seeing 1.46-1.66
        177367, 413973, 413971,
        # r, pass 3, seeing 1.33-1.51
        177365, 177366, 743323,

        # i, pass 1, seeing  < 1.04
        615173, 614494, 421500,
        # i, pass 1, seeing  < 1.1
        614500, 615185, 177750,
        # i, pass 1, seeing  < 1.13
        397530, 395315, 397528,
        # i, pass 1, seeing  < 1.17
        621254, 621612, 405294,
        # i, pass 1, seeing  < 1.21
        397556, 395317, 621615,
        # i, pass 1, seeing  < 1.26
        524711, 420727, 621253,
        # i, pass 1, seeing  < 1.3
        621251, 177072, 177732,
        # i, pass 2, seeing  < 1.36
        177073, 177734, 621250,
        # i, pass 2, seeing  < 1.49
        177122, 177729, 743324,
        # i, pass 2, seeing  < 1.79
        177121, 453887, 413976,

        # z, pass 1, seeing 0.9
        500317, 500316, 514594,
        # z, pass 1, seeing 0.97
        514292, 514605, 514604,
        # z, pass 1, seeing 1.03
        514886, 431104, 514892,
        # z, pass 1, seeing 1.07
        617212, 617198, 707220,
        # z, pass 1, seeing 1.13
        707617, 395319, 514277,
        # z, pass 1, seeing 1.20
        397558, 395347, 515440,
        # z, pass 1, seeing 1.25
        514276, 413981, 707771,
        # z, pass 2, seeing 1.33
        514275, 176844, 525433,
        # z, pass 2, seeing ~1.5
        413978, 176853, 453883,
        # z, pass 2, seeing ~1.55-1.8
        176846, 525634, 525633,
    ]
    
    # Select only our exposures from table E.
    I = []
    for e in exposures:
        print('expnum', e)
        I.append(np.flatnonzero(E.expnum == e)[0])
    I = np.array(I)
    E = E[I]
    print('Cut to', len(E), 'exposures')

# Here's the main code that builds the subsets of exposures.
sets = []
for iset in range(100):
    print('-------------------------------------------')

    thisset = []
    used_expnums = []
    
    for band in bands:
        gotband = False

        B = E[E.filter == band]
        print(len(B), 'exposures in', band, 'band')

        # Compute target detection inverse-variance
        Nsigma = 5.
        sig = NanoMaggies.magToNanomaggies(target[band]) / Nsigma
        targetiv = 1./sig**2

        # What fraction of the exposure time do we want an image to contribute?
        maxfrac = 0.34
        maxiv = maxfrac * targetiv
        # The total detection inverse-variance we have accumulated
        detiv = 0.
        for i in range(len(B)):
            exp = B[i]
            print('Image', exp.expnum, 'propid', exp.propid, 'exptime',
                  exp.exptime, 'seeing', exp.seeing)
            # Grab the individual CCDs in this exposure.
            ccds = C[C.expnum == exp.expnum]
            print(len(ccds), 'CCDs in this exposure')

            # Depth (in detection inverse-variance) of this exposure
            thisdetiv = 1. / (ccds.sig1 / ccds.galnorm)**2
            print('  mean detiv', np.mean(thisdetiv),
                  '= fraction of target: %.2f' % (np.mean(thisdetiv)/ targetiv))

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
            #print('  -> depths:', -2.5 * (np.log10(5./np.sqrt(thisdetiv)) - 9))

            # Add this exposure's CCDs to the set.
            detiv += np.mean(thisdetiv)
            thisset.append(ccds)
            used_expnums.append(exp.expnum)
            # Have we hit our depth target?
            if detiv > targetiv:
                gotband = True
                break
        if not gotband:
            break
    if not gotband:
        break

    # Create the table of CCDs in this subset.
    Cset = merge_tables(thisset)
    Cset.camera = np.array([c + '+noise' for c in Cset.camera])
    sets.append(Cset)

    # Drop the exposures that were used in this subset.
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
