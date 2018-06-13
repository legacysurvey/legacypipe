from astrometry.libkd.spherematch import *
from astrometry.util.fits import *
import numpy as np
from astrometry.util.starutil_numpy import *
from astrometry.util.plotutils import *
from glob import glob
from collections import Counter

'''This script was run after running the depth-cut.py code in this
directory on all bricks in DR7, using a CCDs file that had been cut on
the early_decam data (plus changes detailsed below), and depth-cut.py
code at commit 3c1d83e, ie, including adding DECaLS data first.

(then at commit 385902a)

Launched via:
qdo launch depth 256 --cores_per_worker 1 --walltime=30:00 --batchqueue regular --keep_env --script ./legacyanalysis/depthcut.sh

'''

'''In order to use the faster approximate WCS and pre-computed
galnorms, we use a kd-tree built from the CCDs-annotated file, cut for
ccd_cuts==0;

A = fits_table('/global/cscratch1/sd/dstn/dr7-depthcut-input/ccds-annotated-dr7.fits.gz')
A.cut(A.ccd_cuts == 0)
# Patch with the original CCDs "sig1" values.
C = fits_table('/global/cscratch1/sd/dstn/dr7-depthcut-input/survey-ccds-dr7.fits.gz')
sig1s = dict([((expnum,ccdname.strip()), sig1) for expnum,ccdname,sig1 in zip(C.expnum, C.ccdname, C.sig1)])
A.sig1 = np.array([sig1s.get((expnum,ccdname.strip())) for expnum,ccdname in zip(A.expnum, A.ccdname)]).astype(np.float32)
# Drop extraneous columns to keep the file size down
for col in 'good_region ra0 dec0 ra1 dec1 ra2 dec2 ra3 dec3 dra ddec ra_center dec_center meansky stdsky maxsky minsky pixscale_mean pixscale_std pixscale_max pixscale_min psfnorm_mean psfnorm_std psf_mx2 psf_my2 psf_mxy psf_a psf_b psf_theta psf_ell humidity outtemp tileid tilepass tileebv plver ebv decam_extinction wise_extinction psfdepth galdepth gausspsfdepth gaussgaldepth'.split(' '):
    if col in A.get_columns():
        A.delete_column(col)
A.writeto('/global/cscratch1/sd/dstn/ccds.fits')
# Note, running this from the jupyter notebook server machine doesn't seem to work correctly!!
cmd = 'startree -i /global/cscratch1/sd/dstn/ccds.fits -o /global/cscratch1/sd/dstn/dr7-depthcut-input/survey-ccds-dr7-ann.kd.fits -P -k -n ccds -T'
os.system(cmd)

'''



# Martin's list of number of CCDs per brick.
f = open('nccds.dat')
nccds = {}
for line in f.readlines():
    words = line.strip().split(' ')
    nccds[words[0]] = int(words[1])

B = fits_table('survey-bricks.fits.gz')
B.nccds = np.zeros(len(B), int)
for i,b in enumerate(B.brickname):
    try:
        B.nccds[i] = nccds[b]
    except KeyError:
        pass

I, = np.nonzero(B.nccds)
B.cut(I)
print(len(B), 'bricks touching CCDs')

B.nkept = np.zeros(len(B), int)
B.exists = np.zeros(len(B), bool)
brickccds = {}
for i,brickname in enumerate(B.brickname):
    brickname = B.brickname[i]
    fn = 'depthcuts/%s/ccds-%s.fits' % (brickname[:3], brickname)
    # ARRRgh, due to a bug (fixed in 7a4e51e), the ccds-BBB.fits table was instead written to depth-BBB-z.fits.
    #fn = 'depthcuts/%s/depth-%s-z.fits' % (brickname[:3], brickname)
    if i % 1000 == 0:
        print(fn)
    if not os.path.exists(fn):
        continue
    B.exists[i] = True
    try:
        D = fits_table(fn)
    except:
        print('Failed', fn)
        import traceback
        traceback.print_exc()
        continue
    B.nkept[i] = np.sum(D.passed_depth_cut)
    brickccds[brickname] = (D.expnum, D.ccdname, D.passed_depth_cut)

print(np.sum(B.exists), 'brick files exist')
#plt.scatter(B.ra, B.dec, c=B.nkept, s=3);

# Find all CCDs that pass the depth cut in any brick.
passccds = set()
for brick in B.brickname[B.exists]:
    enums,ccdnames,passed = brickccds[brick]
    passccds.update([(e,c.strip()) for e,c in zip(enums[passed], ccdnames[passed])])
print('CCDs that pass a depth cut:', len(passccds))

# Take the union of CCDs that pass depth cut in any brick
B.nunion = np.zeros(len(B), int)
I, = np.nonzero(B.exists)
for i,brick in zip(I, B.brickname[I]):
    enums,ccdnames,passed = brickccds[brick]
    n = sum([(e,c.strip()) in passccds for e,c in zip(enums, ccdnames)])
    B.nunion[i] = n

# plt.scatter(B.ra, B.dec, c=B.nunion, s=3, vmax=200)
# plt.xlim(360,0)
# plt.title('CCDs per brick, after depth cut')
# plt.colorbar();
# plt.savefig('depthcut.png');

print('Max number of CCDs in union:', max(B.nunion), 'vs per-brick:', max(B.nkept))

# from legacypipe.survey import LegacySurveyData
# survey = LegacySurveyData('/global/cscratch1/sd/dstn/dr7-depthcut-input/')
# C = survey.get_ccds_readonly()
# new_ccd_cuts = C.ccd_cuts + (0x4000 * 
#     np.array([(cuts == 0) and ((e,c) not in passccds)
#               for e,c,cuts in zip(C.expnum, C.ccdname, C.ccd_cuts)]))

C2 = fits_table('/global/cscratch1/sd/dstn/dr7-depthcut-input/survey-ccds-dr7.fits.gz')
C2.ccd_cuts = C2.ccd_cuts + (0x4000 * 
    np.array([(cuts == 0) and ((e,c.strip()) not in passccds)
              for e,c,cuts in zip(C2.expnum, C2.ccdname, C2.ccd_cuts)]))
K, = np.nonzero(C2.propid == '2014B-0404')
print(len(K), 'DECaLS')
print('DECaLS cut by depth cut:', Counter(C2.ccd_cuts[K] == 0x4000))
C2.writeto('/global/cscratch1/sd/dstn/dr7-depthcut/survey-ccds-dr7.fits.gz')

A = fits_table('/global/cscratch1/sd/dstn/dr7-depthcut-input/ccds-annotated-dr7.fits.gz')
A.ccd_cuts = A.ccd_cuts + (0x4000 * 
    np.array([(cuts == 0) and ((e,c.strip()) not in passccds)
              for e,c,cuts in zip(A.expnum, A.ccdname, A.ccd_cuts)]))
A.writeto('/global/cscratch1/sd/dstn/dr7-depthcut/ccds-annotated-dr7.fits.gz')

C2[C2.ccd_cuts == 0].writeto('/global/cscratch1/sd/dstn/ccds.fits')
cmd = 'startree -i /global/cscratch1/sd/dstn/ccds.fits -o /global/cscratch1/sd/dstn/dr7-depthcut-input/survey-ccds-dr7-ann.kd.fits -P -k -n ccds -T'
print(cmd)
os.system(cmd)

