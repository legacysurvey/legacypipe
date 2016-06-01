from __future__ import print_function
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
