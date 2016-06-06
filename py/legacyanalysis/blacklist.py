from __future__ import print_function
import sys
from glob import glob
import pylab as plt
import numpy as np
from legacypipe.common import *
from astrometry.util.plotutils import *

ps = PlotSequence('black')
survey = LegacySurveyData()
ccds = survey.get_ccds_readonly()
propids = np.unique(ccds.propid)
print(len(propids), 'propids')
bands = 'grz'

pa = dict(range=((-50,50),(-6,8)), nbins=(400,80))
phi = 99


# check whether eBOSS-ELG bricks include blacklisted propids
I = survey.photometric_ccds(ccds)
ccds.cut(I)
print('Cut to', len(ccds), 'photometric')
propids = np.unique(ccds.propid)
print(len(propids), 'propids')

expnumtopropid = dict(zip(ccds.expnum, ccds.propid))

I = survey.apply_blacklist(ccds)
okccds = ccds[I]
okpropids = np.unique(okccds.propid)
print(len(okpropids), 'propids after applying blacklist')
badpropids = set(propids) - set(okpropids)
badexpnums = set(ccds.expnum) - set(okccds.expnum)
print('Bad propids:', badpropids)
print('Bad exposure numbers:', len(badexpnums))

# eBOSS-ELG CCDs tables
efn = 'eboss-ccds.fits'
if not os.path.exists(efn):
    eccds = []
    fns = glob('dr3/coadd/[03]*/*/*-ccds.fits')
    fns.sort()
    print(len(fns), 'CCDs tables')
    for fn in fns:
        try:
            T = fits_table(fn, columns=['expnum','ccdname','ra','dec','exptime', 'expid', 'propid'])
            bad = set(T.expnum).intersection(badexpnums)
            print(len(T), 'from', fn, '-> bad', len(bad), bad)
            for expnum in bad:
                print('  expnum', expnum, '-> propid', expnumtopropid[expnum])
    
            eccds.append(T)
        except:
            print('Failed to read', fn)
            import traceback
            traceback.print_exc()
    
    eccds = merge_tables(eccds)
    i,I = np.unique(eccds.expid, return_index=True)
    eccds.cut(I)
    eccds.writeto(efn)
else:
    eccds = fits_table(efn)

print(len(eccds), 'unique CCDs in eBOSS-ELG area')
# wrap
eccds.ra += (-360 * (eccds.ra > 180))

uid = np.unique(eccds.propid)
print('Propids:', uid)

H,xe,ye = plothist(eccds.ra, eccds.dec, weights=eccds.exptime, phi=phi, **pa)
plt.title('CCD Exposure time')
ps.savefig()
tmax = np.percentile(H.ravel(), phi)

H,xe,ye = plothist(eccds.ra, eccds.dec, phi=phi, **pa)
plt.title('Number of CCDs')
ps.savefig()
#mx = np.percentile(H.ravel(), phi)

pt = pa.copy()
pt.update(imshowargs=dict(vmax=tmax))

for pid in uid:
    I = np.flatnonzero(eccds.propid == pid)
    print(len(I), 'from propid', pid)

    plt.clf()
    H,xe,ye = plothist(eccds.ra[I], eccds.dec[I], weights=eccds.exptime[I], **pt)
    plt.title('CCD Exposure time: Propid %s' % pid)
    ps.savefig()

    I = np.flatnonzero(eccds.propid != pid)
    plt.clf()
    H,xe,ye = plothist(eccds.ra[I], eccds.dec[I], weights=eccds.exptime[I], **pt)
    plt.title('CCD Exposure time: Without propid %s' % pid)
    ps.savefig()


# eBOSS-ELG catalogs
ecats = []
fns = glob('dr3/tractor/[03]*/tractor-*.fits')
fns.sort()
print(len(fns), 'tractor catalogs')
for fn in fns:
    try:
        T = fits_table(fn, columns=['ra','dec','brick_primary'])
        ecats.append(T)
    except:
        print('Failed to read', fn)
        import traceback
        traceback.print_exc()
ecats = merge_tables(ecats)

ecats.ra += (-360 * (ecats.ra > 180))
H,xe,ye = plothist(ecats.ra, ecats.dec, phi=phi, **pa)
plt.title('Number of sources')
ps.savefig()
#plt.savefig('black-02.png')
          
sys.exit(0)





pa = dict(range=((0,360),(-20,40)), nbins=(360,60))
pa0 = pa.copy()

plothist(ccds.ra, ccds.dec, **pa)
plt.title('CCDs')
plt.axis('scaled')
ps.savefig()

phi = 99
H,xe,ye = plothist(ccds.ra, ccds.dec, weights=ccds.exptime, phi=phi, **pa)
#plt.axis('scaled')
#plt.axis([0,360,-20,40])
plt.title('CCD Exposure time')
ps.savefig()
mx = np.percentile(H.ravel(), phi)
#pa.update(imshowargs=dict(vmax=mx), docolorbar=False)

phi = 99
H,xe,ye = plothist(ccds.ra, ccds.dec, phi=phi, docolorbar=False, **pa)
plt.title('Number of CCDs')
ps.savefig()
mx = np.percentile(H.ravel(), phi)
pa.update(imshowargs=dict(vmax=mx), docolorbar=False)

I = survey.photometric_ccds(ccds)
ccds.cut(I)
H,xe,ye = plothist(ccds.ra, ccds.dec, phi=phi, **pa)
plt.title('Number of CCDs -- Photometric')
ps.savefig()

I = survey.apply_blacklist(ccds)
H,xe,ye = plothist(ccds.ra[I], ccds.dec[I], phi=phi, **pa)
plt.title('Number of CCDs -- Photometric & Not Blacklisted')
ps.savefig()

H,xe,ye = plothist(ccds.ra[I], ccds.dec[I], weights=ccds.exptime[I], phi=phi,
                   **pa0)
plt.title('Exptime -- Photometric & Not Blacklisted')
ps.savefig()


ntotal = []
for propid in propids:
    I = np.flatnonzero((ccds.propid == propid))
    ntotal.append(len(I))
J = np.argsort(-np.array(ntotal))
    
for propid in propids[J]:
    # for band in bands:
    #     I = np.flatnonzero((ccds.propid == propid) * (ccds.filter == band))
    #     print(len(I), 'from propid', propid, 'in band', band)
    #     pa = dict(range=((0,360),(-20,40)), bins=(360,60))
    #     plothist(ccds.ra, ccds.dec, weights=ccds.exptime, **pa)
    #     plt.title('Exposure time: propid %s, band %s' % (propid, band))
    #     ps.savefig()
    print()
    I = np.flatnonzero((ccds.propid == propid))
    print(len(I), 'from propid', propid)
    #plothist(ccds.ra[I], ccds.dec[I], weights=ccds.exptime[I], **pa)
    #plt.title('Exposure time: propid %s' % (propid))
    plothist(ccds.ra[I], ccds.dec[I], **pa)
    plt.title('CCDs: propid %s' % (propid))
    ps.savefig()

    E,J = np.unique(ccds.expnum[I], return_index=True)
    J = I[J]
    print('Prop ID', propid, ':', len(E), 'exposures')
    for e in ccds[J]:
        print('  expnum', e.expnum, 'exptime', e.exptime, 'RA,Dec (%.1f,%.1f)' % (e.ra_bore, e.dec_bore), 'filter', e.filter, 'object', e.object)
