from __future__ import print_function
from astrometry.util.fits import *
from collections import Counter
import pylab as plt

from legacypipe.survey import LegacySurveyData

survey = LegacySurveyData()
ann = survey.get_annotated_ccds()
print('Got', len(ann), 'annotated CCDs')
ann.about()

# build mapping from expnum,ccdname to index in ann table.
annmap = dict([((e,c.strip()),i) for i,(e,c) in enumerate(zip(ann.expnum, ann.ccdname))])

ccds = fits_table('/global/cscratch1/sd/kaylanb/dr5_zpts/survey-ccds-legacypipe-hdufix-45455-nocuts.fits.gz')
print('Read', len(ccds), 'CCDs')

print('Cameras:', np.unique(ccds.camera))

Iann = np.array([annmap.get((e,c.strip()), -1) for e,c in zip(ccds.expnum, ccds.ccdname)])
print(np.sum(Iann >= 0), 'CCD entries are found in annotated CCDs table')
print('10 most common indices:', Counter(Iann).most_common(10))

assert(np.all(Iann >= 0))

keep = fits_table('depth-cut-kept-ccds.fits')
print('Read', len(keep), 'CCDs to keep')
keep = set(zip(keep.expnum, [c.strip() for c in keep.ccdname]))
print('Keep:', len(keep))

#ccds.depth_cut_ok = np.zeros(len(ccds), bool)
ccds.depth_cut_ok = np.array([((e,c.strip()) in keep) for e,c in zip(ccds.expnum, ccds.ccdname)])
#print('Depth_cut_ok: unique', np.unique(ccds.depth_cut_ok), 'True:', np.sum(ccds.depth_cut_ok == True), 'False:', np.sum(ccds.depth_cut_ok == False))
print(Counter(ccds.depth_cut_ok).most_common())

zpok = np.flatnonzero(np.isfinite(ccds.ccdzpt) * np.isfinite(ccds.zpt) * np.isfinite(ccds.fwhm))
zpok = zpok[np.flatnonzero((ccds.ccdzpt[zpok] > 0) * (ccds.zpt[zpok] > 0) * (ccds.fwhm[zpok] > 0))]
ccds.has_zeropoint = np.zeros(len(ccds), bool)
ccds.has_zeropoint[zpok] = True

annccds = ccds.copy()
cols = annccds.columns()
for c in ann.get_columns():
    if c in cols:
        continue
    print('Column', c)
    annccds.set(c, ann.get(c)[Iann])
annccds.writeto('ccds-annotated.fits')
del annccds

print('Number of CCDs with zeropoints:', Counter(ccds.has_zeropoint).most_common())

print('ccdzpt finite:', Counter(np.isfinite(ccds.ccdzpt)))
print('ccdzpt finite and > 0:', Counter(np.isfinite(ccds.ccdzpt) * (ccds.ccdzpt > 0)).most_common())
print('zpt > 0:', Counter((ccds.zpt > 0)).most_common())
print('fwhm > 0:', Counter((ccds.fwhm > 0)).most_common())

print('Number of CCDs where depth_cut_ok but no zeropoint:', np.sum(ccds.depth_cut_ok * (ccds.has_zeropoint == False)))

I = np.flatnonzero(ccds.depth_cut_ok * (ccds.has_zeropoint == False))
print('Exposure numbers:', len(np.unique(ccds.expnum[I])), np.unique(ccds.expnum[I]))

print('Depth cut but no zeropoint:')
print('  ccdzpt finite:', Counter(np.isfinite(ccds.ccdzpt[I])))
print('  ccdzpt finite and > 0:', Counter(np.isfinite(ccds.ccdzpt[I]) * (ccds.ccdzpt[I] > 0)))
print('  zpt finite:', Counter(np.isfinite(ccds.zpt[I])))
print('  zpt finite and > 0:', Counter(np.isfinite(ccds.zpt[I]) * (ccds.zpt[I] > 0)))
print('  fwhm finite:', Counter(np.isfinite(ccds.fwhm[I])))
print('  fwhm finite and > 0:', Counter(np.isfinite(ccds.fwhm[I]) * (ccds.fwhm[I] > 0)))

plt.clf()
plt.plot(ccds.ra, ccds.dec, 'b.')
plt.title('DR5: All CCDs (%i)' % len(ccds))
plt.savefig('radec1.png')
ax = [195,205,-20,10]
plt.axis(ax)
plt.savefig('radec1b.png')

plt.clf()
plt.plot(ccds.ra, ccds.dec, 'b.')
plt.plot(ccds.ra[I], ccds.dec[I], 'r.')
plt.title('DR5: CCDs that pass depth cut but have no zeropoints (%i)' % len(I))
plt.savefig('radec2.png')
plt.axis(ax)
plt.savefig('radec2b.png')

plt.clf()
plt.plot(ccds.ra[ccds.has_zeropoint == False], ccds.dec[ccds.has_zeropoint == False], 'r.')
plt.title('DR5: CCDs missing zeropoints (%i)' % (np.sum(ccds.has_zeropoint == False)))
plt.savefig('radec3.png')
plt.axis(ax)
plt.savefig('radec3b.png')


ccds.cut(ccds.has_zeropoint * ccds.depth_cut_ok)
print('Cut to', len(ccds), 'with zeropoints that pass depth cut')
ccds.writeto('/tmp/ccds.fits')

cmd = 'startree -i /tmp/ccds.fits -o /tmp/kd.fits -P -k -n ccds'
print(cmd)
rtn = os.system(cmd)
assert(rtn == 0)
outfn = 'survey-ccds-dr5.kd.fits'
cmd = 'fitsgetext -i /tmp/kd.fits -o %s -e 0 -e 6 -e 1 -e 2 -e 3 -e 4 -e 5' % outfn
print(cmd)
rtn = os.system(cmd)
assert(rtn == 0)
print('Wrote', outfn)

