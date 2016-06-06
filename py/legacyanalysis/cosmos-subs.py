from __future__ import print_function
from astrometry.util.fits import *
import numpy as np
import pylab as plt

from legacypipe.common import *
from legacypipe.decam import DecamImage

bands = 'grz'

cfn = 'cosmos-ccds.fits'
if not os.path.exists(cfn):
    #survey = LegacySurveyData(version='dr2')
    survey = LegacySurveyData()
    
    C = survey.get_annotated_ccds()
    print(len(C), 'annotated CCDs')

    C.cut(np.hypot(C.ra_bore - 150, C.dec_bore - 2.2) < 1.)
    print(len(C), 'CCDs on COSMOS')

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
    
nil,I = np.unique(C.expnum, return_index=True)
E = C[I]
print(len(E), 'exposures')
E.galnorm = E.galnorm_mean

# Target depths (90th percentile), for 5-sigma galaxy profile
target = dict(g=24.0, r=23.4, z=22.5)

isdecals = np.array([p == '2014B-0404' for p in E.propid])
print('Is DECaLS:', np.unique(isdecals), sum(isdecals), len(E))
# Lexsort doesn't seem to work with only a single boolean column; add dumb arange
E.cut(np.lexsort((np.arange(len(E)), E.filter, np.logical_not(isdecals))))

E.index = np.arange(len(E))
E.passnum = np.zeros(len(E), np.uint8)
E.depthfraction = np.zeros(len(E), np.float32)

zp0 = DecamImage.nominal_zeropoints()
# HACK -- copied from obsbot
kx = dict(g = 0.178,
          r = 0.094,
          z = 0.060,)

for band in bands:
    B = E[E.filter == band]
    B.cut(np.argsort(B.seeing))
    #print(len(B), 'exposures in', band, 'band')

    Nsigma = 5.
    sig = NanoMaggies.magToNanomaggies(target[band]) / Nsigma
    targetiv = 1./sig**2
    
    for exp in B:
        thisdetiv = 1. / (exp.sig1 / exp.galnorm)**2
        # Which pass number would this image be assigned?
        trans = 10.**(-0.4 * (zp0[band] - exp.ccdzpt
                              - kx[band]*(exp.airmass - 1.)))
        #print('Transparency', trans)
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

        #print('  exp', exp.expnum, 'seeing %.3f' % exp.seeing, 'exptime %3.0f' % exp.exptime, 'propid', exp.propid, 'fraction of depth: %.2f' % depthfrac, 'sig1 %.4f' % exp.sig1, 'pass', E.passnum[exp.index], ('X' if depthfrac < 0.34 else ''))


for band in bands:
    B = E[E.filter == band]
    B.cut(np.lexsort((B.seeing, (B.depthfraction < 0.34), B.passnum)))
    print(len(B), 'exposures in', band, 'band')
    for exp in B:
        print('  e', exp.expnum, 'see %.2f' % exp.seeing,
              't %3.0f' % exp.exptime, 'pid', exp.propid,
              'f.depth: %.2f' % exp.depthfraction, #'sig1 %.4f' % exp.sig1,
              'pass', exp.passnum, ('X' if exp.depthfraction < 0.34 else ''))

        
sets = []
for iset in xrange(100):
    print('-------------------------------------------')
    thisset = []
    addnoise = []
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
            print('Image', exp.expnum, 'propid', exp.propid, 'exptime', exp.exptime, 'seeing', exp.seeing)
            thisdetiv = 1. / (exp.sig1 / exp.galnorm)**2
            print('  detiv', thisdetiv, '= fraction of target: %.2f' % (thisdetiv / targetiv))

            maxiv = maxfrac * targetiv
            if thisdetiv > maxiv:
                #addnoise = 
                targetsig1 = exp.sig1 * np.sqrt(thisdetiv / maxiv)
                addsig1 = np.sqrt(max(0, targetsig1**2 - exp.sig1**2))
                addnoise.append(addsig1)
                newiv = 1. / (np.hypot(exp.sig1, addsig1) / exp.galnorm)**2
                #print 'Adding', addsig1, 'we will get detiv', newiv, 'vs', maxiv
                print('  adding factor %.2f more noise' % (addsig1/exp.sig1))
                thisdetiv = maxiv
            else:
                addnoise.append(0.)

            #thisdetiv = min(thisdetiv, maxfrac * targetiv)
            detiv += thisdetiv
            thisset.append(exp.expnum)
            if detiv > targetiv:
                gotband = True
                break
        if not gotband:
            break
    if not gotband:
        break

    Eset = E[np.array([expnum in thisset for expnum in E.expnum])]
    print('Cut to', len(E), 'exposures in set')

    Eset.addnoise = np.array(addnoise)
    #sets.append(thisset)
    #sets.append(Eset)

    expnoise = dict(zip(Eset.expnum, Eset.addnoise))
    Cset = C[np.array([expnum in thisset for expnum in C.expnum])]
    Cset.addnoise = np.array([expnoise[e] for e in Cset.expnum]).astype(np.float32)
    Cset.camera = np.array([c + '+noise' for c in Cset.camera])
    sets.append(Cset)
    
    E.cut(np.array([expnum not in thisset for expnum in E.expnum]))
    print('Cut to', len(E), 'remaining exposures')


print('Got', len(sets), 'sets of exposures')

for i,C in enumerate(sets):
    C.writeto('cosmos-ccds-sub%i.fits' % i)
    C.subset = np.array([i] * len(C)).astype(np.uint8)
    
C = merge_tables(sets)
C.writeto('cosmos-ccds.fits')

#for i,E in enumerate(sets):
#    E.writeto('cosmos-subset-%i.fits' % i)

