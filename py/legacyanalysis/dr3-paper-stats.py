from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
import pylab as plt
import numpy as np
from astrometry.util.fits import *
from astrometry.libkd.spherematch import match_radec
import fitsio

T1 = fits_table('ccds-annotated-decals.fits.gz')
T2 = fits_table('ccds-annotated-extra.fits.gz')
T3 = fits_table('ccds-annotated-nondecals.fits.gz')

T = merge_tables([T1,T2,T3], columns='fillzero')
print(len(T), 'CCDs')
T.cut(T.photometric)
T.cut(T.blacklist_ok)
print(len(T), 'good CCDs')

E = T[np.unique(T.expnum, return_index=True)[1]]
print(len(E), 'exposures')

D = E[E.propid == '2014B-0404']
print(len(D), 'DECaLS exposures')

ND = E[E.propid != '2014B-0404']
print(len(ND), 'Non-DECaLS exposures')

plt.figure(figsize=(4,3))
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.98, top=0.99)

ccmap = dict(g='g', r='r', z='m')
lwmap = dict(g=1, r=2, z=3)
amap = dict(g=1, r=0.5, z=0.3)

plt.clf()
for band in 'grz':
    ha = dict(range=(0,3.0), bins=60, histtype='step', normed=True)
    I = np.flatnonzero(E.filter == band)
    print(len(I), 'exposures in band', band)
    plt.hist(E.seeing[I], color=ccmap[band], lw=lwmap[band], alpha=amap[band], label=band, **ha)
    # I = np.flatnonzero(D.filter == band)
    # plt.hist(D.seeing[I], color=ccmap[band], lw=lwmap[band], alpha=amap[band], label=band, **ha)
    # I = np.flatnonzero(ND.filter == band)
    # plt.hist(ND.seeing[I], color=ccmap[band], lw=lwmap[band], alpha=amap[band], linestyle='--', **ha)
plt.legend()
plt.xticks(np.arange(0, 3, 0.5))
plt.yticks([])
plt.ylabel('Relative frequency')
plt.xlim(0.5, 2.5)
plt.xlabel('Seeing (arcsec)')
plt.savefig('seeing.pdf')


plt.clf()
for band in 'grz':
    ha = dict(range=(0,3.0), bins=60, histtype='step', normed=True)
    I = np.flatnonzero(E.filter == band)
    plt.hist(E.airmass[I], color=ccmap[band], lw=lwmap[band], alpha=amap[band], label=band, **ha)
plt.legend()
plt.xticks(np.arange(0, 3, 0.5))
plt.yticks([])
plt.ylabel('Relative frequency')
plt.xlim(0.9, 2.5)
plt.xlabel('Airmass')
plt.savefig('airmass.pdf')


plt.clf()
for band in 'grz':
    ha = dict(range=(16.5, 22.5), bins=60, histtype='step', normed=True)
    I = np.flatnonzero(E.filter == band)
    plt.hist(E.ccdskymag[I], color=ccmap[band], lw=lwmap[band], alpha=amap[band], label=band, **ha)
plt.legend(loc='upper left')
plt.xticks(np.arange(16, 23, 1.))
plt.yticks([])
plt.ylabel('Relative frequency')
plt.xlim(16.5, 22.5)
# Confirmed that these are mag/arcsec^2 based on equality with obsbot's skybr.
plt.xlabel('Sky Brightness (mag / arcsec$^2$)')
plt.savefig('sky.pdf')



plt.clf()
lo,hi = 20.75, 24.75
for band in 'grz':
    ha = dict(range=(lo,hi), bins=40, histtype='step', normed=True)
    I = np.flatnonzero(E.filter == band)
    iband = dict(g=1, r=2, z=4)[band]
    plt.hist(E.galdepth[I] - E.decam_extinction[I,iband], color=ccmap[band], lw=lwmap[band], alpha=amap[band], label=band, **ha)
plt.legend(loc='upper left')
#plt.xticks(np.arange(21, 25.1, 0.5))
plt.yticks([])
plt.ylabel('Relative frequency')
plt.xlim(lo,hi)
plt.xlabel('Depth (5-$\sigma$, galaxy profile, extinction corrected)')
plt.savefig('depth.pdf')

sys.exit(0)

dates = np.unique(D.date_obs)
print('DECaLS date range:', dates[0], dates[-1])
print('DECaLS expnum range:', D.expnum.min(), D.expnum.max())
lastexp = D.expnum.max()

obs = fits_table('obstatus/decam-tiles_obstatus.fits')
print(len(obs), 'tiles')
obs.cut((obs.in_desi == 1) * (obs.in_des == 0))
print(len(obs), 'tiles in DESI and in DES')
print(obs.dec.max(), 'max Dec')
print(obs.dec.min(), 'min Dec')

print(np.sum((obs.g_done == 1) * (obs.g_expnum > 0) * (obs.g_expnum <= lastexp)), 'g tiles done as of DR3')
print(np.sum((obs.r_done == 1) * (obs.r_expnum > 0) * (obs.r_expnum <= lastexp)), 'r tiles done as of DR3')
print(np.sum((obs.z_done == 1) * (obs.z_expnum > 0) * (obs.z_expnum <= lastexp)), 'z tiles done as of DR3')

# cd ~/observing
# svn up -r 1188
# cp obstatus/decam-tiles_obstatus.fits ~/legacypipe/py/decam-tiles_obstatus-2016-03-02.fits

print('Svn rev 1188 tiles:')
obs = fits_table('decam-tiles_obstatus-2016-03-02.fits')

obs.plotra = obs.ra + (-360 * (obs.ra > 300))

print(len(obs), 'tiles')
#desiobs = obs[(obs.in_desi == 1)]
desobs = obs[(obs.in_desi == 1) * (obs.in_des == 1)]
obs.cut((obs.in_desi == 1) * (obs.in_des == 0))

print(len(obs), 'tiles in DESI and in DES')
print(obs.dec.max(), 'max Dec')
print(obs.dec.min(), 'min Dec')
obs.cut(obs.dec < 35.)
print(len(obs), 'with Dec < 35')
print(obs.dec.max(), 'max Dec')
print(np.sum((obs.g_done == 1) * (obs.g_expnum > 0) * (obs.g_expnum <= lastexp)), 'g tiles done as of DR3')
print(np.sum((obs.r_done == 1) * (obs.r_expnum > 0) * (obs.r_expnum <= lastexp)), 'r tiles done as of DR3')
print(np.sum((obs.z_done == 1) * (obs.z_expnum > 0) * (obs.z_expnum <= lastexp)), 'z tiles done as of DR3')

print('Largest expnum listed:', max([obs.g_expnum.max(), obs.r_expnum.max(), obs.z_expnum.max()]))

plt.figure(figsize=(4,3))
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.99, top=0.99)

for passnum in [1,2,3,4]:
    for band in 'grz':
        pn = min(3, passnum)

        I = np.flatnonzero(obs.get('pass') == pn)
        ntiles = len(I)

        plt.clf()
        plt.plot(obs.plotra[I], obs.dec[I], 'o', color='k', alpha=0.2,
                 mec='none', mfc='k', ms=2, label='All tiles')
        I = np.flatnonzero(desobs.get('pass') == pn)
        plt.plot(desobs.plotra[I], desobs.dec[I], 'o', color='k', alpha=0.8,
                 mec='none', mfc='k', ms=2, label='Tiles in DES')

        if passnum < 4:
            I = np.flatnonzero((obs.get('pass') == passnum) *
                               (obs.get('%s_done' % band) == 1))
            ndone = len(I)
            plt.plot(obs.plotra[I], obs.dec[I], 'o', mec='k', mfc='k', ms=4,
                     label='Tiles finished (%s pass %i)' % (band, passnum))

            pdone = (100. * ndone / float(ntiles))
            print('Pass', passnum, 'band', band, ':', ndone, 'of', ntiles,
                  ': %.1f' % pdone)
            plt.text(80, 30, '%.1f' % pdone + '\\% done')
            
        else:
            I = np.flatnonzero(ND.filter == band)
            print(len(I), 'non-DECaLS exposures in band', band)
            # II,J,d = match_radec(ND.ra_bore[I], ND.dec_bore[I],
            #                      desiobs.ra, desiobs.dec, 2., nearest=True)
            # print(len(II), 'matches to DECaLS tiles')
            # I = I[II]
            plt.plot(ND.ra_bore[I], ND.dec_bore[I],'o',mec='k', mfc='k', ms=4,
                     label='Non-DECaLS %s' % band)
                
        plt.xlabel('RA (deg)')
        plt.ylabel('Dec (deg)')
        #plt.axis([360,0,-22,36])
        #plt.xticks(np.arange(0, 361, 60))
        xt = np.arange(-60, 361, 60)
        plt.xticks(xt, ['%i' % ((x + 360)%360) for x in xt])
        plt.axis([300,-60,-22,36])
        plt.legend(loc='lower left')
        plt.savefig('done-%s-%i.pdf' % (band, passnum))
