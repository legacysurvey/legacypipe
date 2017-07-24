from __future__ import print_function
from astrometry.util.fits import *
from collections import Counter
import pylab as plt

from legacypipe.survey import LegacySurveyData

'''
This script takes a legacy-zeropoints CCDs table, plus the
pre-computed list of exposure number + ccdnames of CCDs required to
hit per-brick depth targets, and keeps only the CCDs that are required
to hit our depth targets.

It also reads in a pre-computed annotated-CCDs table and copies the
annotated values to create a new annotated-CCDs table.

You should set LEGACY_SURVEY_DIR to the *old* (ie, IDL zeropoints)
directory; ie, /global/cscratch1/sd/desiproc/dr5.  The results will be
written to the current directory.
'''

survey = LegacySurveyData()

# Read *old* annotated-CCDs tables.
ann = survey.get_annotated_ccds()
print('Got', len(ann), 'annotated CCDs')
ann.about()

# build mapping from expnum,ccdname to index in ann table.
annmap = dict([((e,c.strip()),i) for i,(e,c) in enumerate(zip(ann.expnum, ann.ccdname))])

# Read the *new* zeropoints file.
ccds = fits_table('/global/cscratch1/sd/kaylanb/dr5_zpts/survey-ccds-legacypipe-hdufix-45455-nocuts.fits.gz')
print('Read', len(ccds), 'CCDs')

print('Cameras:', np.unique(ccds.camera))

# Find the row number of each entry in the new zeropoints file in the
# old annotated file (by expnum+ccdname).
Iann = np.array([annmap.get((e,c.strip()), -1) for e,c in zip(ccds.expnum, ccds.ccdname)])
print(np.sum(Iann >= 0), 'CCD entries are found in annotated CCDs table')
# We should have matches for everything.
assert(np.all(Iann >= 0))

# Read in the file of expnum+ccdnames that pass the per-brick depth cuts.
keep = fits_table('/global/homes/d/dstn/legacypipe/py/depth-cut-kept-ccds.fits')
print('Read', len(keep), 'CCDs to keep')
keep = set(zip(keep.expnum, [c.strip() for c in keep.ccdname]))
print('Keep:', len(keep))

# Compute which of the new CCDs pass the depth cut.
ccds.depth_cut_ok = np.array([((e,c.strip()) in keep) for e,c in zip(ccds.expnum, ccds.ccdname)])
print(Counter(ccds.depth_cut_ok).most_common())

# Are the new zeropoints ok? (finite, non-zero)
zpok = np.flatnonzero(np.isfinite(ccds.ccdzpt) * np.isfinite(ccds.zpt) * np.isfinite(ccds.fwhm))
zpok = zpok[np.flatnonzero((ccds.ccdzpt[zpok] > 0) * (ccds.zpt[zpok] > 0) * (ccds.fwhm[zpok] > 0))]
ccds.has_zeropoint = np.zeros(len(ccds), bool)
ccds.has_zeropoint[zpok] = True

# Create the new annotated-CCDs table
annccds = ccds.copy()
cols = annccds.columns()
for c in ann.get_columns():
    if c in cols:
        continue
    print('Column', c)
    annccds.set(c, ann.get(c)[Iann])
annfn = 'ccds-annotated.fits'
annccds.writeto(annfn)
del annccds
print('Wrote', annfn)

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

# Create the kd-tree file.
# First, cut to just the good CCDs (pass the depth cut, have zeropoints)
# and write out that FITS table.
ccds.cut(ccds.has_zeropoint * ccds.depth_cut_ok)
print('Cut to', len(ccds), 'with zeropoints that pass depth cut')
ccds.writeto('/tmp/ccds.fits')

# Now, run the "startree" program, from Astrometry.net (v0.72) to
# create the kd-tree file.
cmd = 'startree -i /tmp/ccds.fits -o /tmp/kd.fits -P -k -n ccds'
print(cmd)
rtn = os.system(cmd)
assert(rtn == 0)
outfn = 'survey-ccds-dr5.kd.fits'

# Now, reorder the HDUs in the KD-tree file, so that the CCDs table
# comes first, and the KD-tree structures follow.
cmd = 'fitsgetext -i /tmp/kd.fits -o %s -e 0 -e 6 -e 1 -e 2 -e 3 -e 4 -e 5' % outfn
print(cmd)
rtn = os.system(cmd)
assert(rtn == 0)
print('Wrote', outfn)

