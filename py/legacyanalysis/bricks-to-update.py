import matplotlib
matplotlib.use('Agg')
import pylab as plt

from astrometry.util.fits import *
import numpy as np

C1 = fits_table('/global/cscratch1/sd/desiproc/dr5-eboss/survey-ccds-eboss-patched.fits.gz')
C2 = fits_table('/global/cscratch1/sd/desiproc/dr5-eboss2/survey-ccds-eboss-morecuts-fwhm.fits.gz')
print(len(C1), 'and', len(C2), 'CCDs')

C1.filter = np.array([f.strip() for f in C1.filter])
C2.filter = np.array([f.strip() for f in C2.filter])

#assert(np.all(C1.expnum == C2.expnum))
#assert(np.all(C1.filter == C2.filter))

from legacypipe.decam import DecamImage

z0 = DecamImage.nominal_zeropoints()
z0 = np.array([z0[f[0]] for f in C1.filter])

C1.phot = ((C1.exptime >= 30) *
           (C1.ccdnmatch >= 20) *
           (np.isfinite(C1.zpt)) *
           (np.isfinite(C1.ccdzpt)) *
           (np.abs(C1.zpt - C1.ccdzpt) <= 0.1) *
           (C1.zpt >= (z0 - 0.5)) *
           (C1.zpt <= (z0 + 0.25)))
print(np.sum(C1.phot), 'CCDs are photometric with old cuts')

z0 = DecamImage.nominal_zeropoints()
z0 = np.array([z0[f[0]] for f in C2.filter])

C2.phot = ((C2.exptime >= 30) *
           (C2.ccdnmatch >= 20) *
           (np.isfinite(C2.zpt)) *
           (np.isfinite(C2.ccdzpt)) *
           (np.abs(C2.zpt - C2.ccdzpt) <= 0.5) *
           (C2.phrms <= 0.1) *
           (C2.zpt >= (z0 - 0.5)) *
           (C2.zpt <= (z0 + 0.25)))
print(np.sum(C2.phot), 'CCDs are photometric with new cuts')

oldphot = dict([((expnum,ccdname),phot) for expnum,ccdname,phot in zip(C1.expnum, C1.ccdname, C1.phot)])
#newphot = dict([((expnum,ccdname),phot) for expnum,ccdname,phot in zip(C1.expnum, C1.ccdname, C1.phot)])

C2.oldphot = np.array([oldphot[(expnum,ccdname)] for expnum,ccdname in zip(C2.expnum, C2.ccdname)])

print(np.sum(C2.phot * np.logical_not(C2.oldphot)), 'CCDs went from non to photometric')
print(np.sum(np.logical_not(C2.phot) * C2.oldphot), 'CCDs went from photometric to non')


bricknames = [b.strip() for b in open('ebossDR5.bricklist').readlines()]
print('Brick names:', bricknames)

from legacypipe.survey import LegacySurveyData, ccds_touching_wcs, wcs_for_brick
survey = LegacySurveyData()

Cchanged = C2[C2.phot != C2.oldphot]
print(len(Cchanged), 'CCDs changed')

plt.clf()
plt.plot(C2.ra, C2.dec, 'k.', alpha=0.2)
plt.plot(Cchanged.ra, Cchanged.dec, 'r.')
plt.title('Changed CCDs')
plt.savefig('ccds.png')

rerun = []
for name in bricknames:
    brick = survey.get_brick_by_name(name)
    wcs = wcs_for_brick(brick)
    I = ccds_touching_wcs(wcs, Cchanged)
    print('Brick', name, ':', len(I), 'CCDs changed photometric cut')
    if len(I):
        rerun.append((name, brick.ra, brick.dec))

plt.plot([r for b,r,d in rerun], [d for b,r,d in rerun], 'b.')
plt.title('Bricks to update')
plt.savefig('bricks.png')


print()
print('Rerun bricks:')
for b,r,d in rerun:
    print(b)





