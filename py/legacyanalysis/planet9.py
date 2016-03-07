from __future__ import print_function

import os

from astrometry.util.fits import fits_table
from astrometry.util.plotutils import *
from astrometry.util.starutil_numpy import *

if __name__ == '__main__':
    ps = PlotSequence('p9')
    
    #ra = np.arange(0, 361, 30)
    ra = np.array([0,   30,  45,  60,   90,  105,  120,  150,   180,
                   210, 240, 270, 300, 330, 360])
    
    #          0   30  (45)  60   90  (105) 120  150   180
    declo = [-40, -25, -20, -15, -5,    0,    5, 15,   20,
             15, -5, -25, -40, -45, -40]

    #          0   30  (45)  60   90  (105) 120  150   180
    dechi = [-20, -15,  -5,   5,  25,   30,  35, 40,   35,
             25, 15, 5, -5, -15, -20]

    D = fits_table('obstatus/decam-tiles_obstatus.fits')
    M = fits_table('obstatus/mosaic-tiles_obstatus.fits')
    # DR2
    C = fits_table('decals/decals-ccds.fits.gz')
    # cori:/global/cscratch1/sd/arjundey/NonDECaLS_DR3/DECam-allframes-2016feb22.fits
    # NonDECaLS - DR3
    N = fits_table('DECam-allframes-2016feb22.fits')
    N.band = np.array([f[0] for f in N.filter])

    for band in ['g','r','z']:
    
        DZ = D[D.get('%s_done' % band) > 0]
        CZ = C[C.filter == band]
        NZ = N[N.band == band]
        Npub = NZ[NZ.releasedate < '2016-03-01']
        NX = NZ[NZ.releasedate >= '2016-03-01']
        if band == 'z':
            MZ = M[M.get('%s_done' % band) > 0]
        else:
            MZ = None
            
        plt.clf()
        leg = []
        pn = plt.plot(NX.ra, NX.dec, 'k.', alpha=0.25)
        leg.append((pn[0], 'NonDECaLS(private)'))
        pn = plt.plot(Npub.ra, Npub.dec, 'r.', alpha=0.5)
        leg.append((pn[0], 'NonDECaLS(DR3)'))
        pd = plt.plot(DZ.ra, DZ.dec, 'm.', alpha=0.5)
        leg.append((pd[0], 'DECaLS'))
        if MZ is not None:
            pm = plt.plot(MZ.ra, MZ.dec, 'g.', alpha=0.5)
            leg.append((pm[0], 'Mosaic'))
        p2 = plt.plot(CZ.ra, CZ.dec, 'b.', alpha=0.5)
        leg.append((p2[0], 'DR2'))

        for rlo,rhi in [(240, 300), (0, 150), (45, 105)]:
            ilo = np.flatnonzero(ra == rlo)[0]
            ihi = np.flatnonzero(ra == rhi)[0]
            sty = dict(lw=3, alpha=0.25)
            plt.plot([rlo,rlo], [declo[ilo],dechi[ilo]], 'k-', **sty)
            plt.plot([rhi,rhi], [declo[ihi],dechi[ihi]], 'k-', **sty)
            plt.plot(ra[ilo:ihi+1], declo[ilo:ihi+1],    'k-', **sty)
            plt.plot(ra[ilo:ihi+1], dechi[ilo:ihi+1],    'k-', **sty)
        
        plt.plot(ra, declo, 'k--')
        plt.plot(ra, dechi, 'k--')
        ax = [360,0,-40,40]
        plt.axvline(45, color='k', linestyle=':')
        plt.axvline(105, color='k', linestyle=':')
        plt.axvline(0, color='k', linestyle='--')
        plt.axvline(120, color='k', linestyle='-')
        plt.axvline(240, color='k', linestyle='--')
        plt.axvline(300, color='k', linestyle='--')
        plt.axis(ax)
        plt.legend((p for p,t in leg), (t for p,t in leg), loc='lower center')
        plt.xlabel('RA (deg)')
        plt.ylabel('Dec (deg)')
        plt.title('Planet Nine -- %s band' % band)

        ps.savefig()
    
    
    A = fits_table('decals/decals-ccds-annotated.fits')
    
    if False:
        for band in ['g','r','z']:
            Ab = A[(A.filter == band) * (A.tilepass > 0)]
            ax = [360, 0, -10, 35]
            plt.clf()
            I = np.flatnonzero(Ab.tilepass == 1)
            plt.plot(Ab.ra[I], Ab.dec[I], 'k.', alpha=0.02)
            plt.xlabel('RA (deg)')
            plt.ylabel('Dec (deg)')
            plt.title('%s band, pass 1' % band)
            plt.axis(ax)
            ps.savefig()
            I = np.flatnonzero(Ab.tilepass == 2)
            plt.plot(Ab.ra[I], Ab.dec[I], 'k.', alpha=0.02)
            plt.title('%s band, pass 1,2' % band)
            plt.axis(ax)
            ps.savefig()
            I = np.flatnonzero(Ab.tilepass == 3)
            plt.plot(Ab.ra[I], Ab.dec[I], 'k.', alpha=0.02)
            plt.title('%s band, pass 1,2,3' % band)
            plt.axis(ax)
            ps.savefig()

    h,w = 300,600
    rgb = np.zeros((h,w,3), np.uint8)
    ax = [360, 0, -10, 35]
    for iband,band in enumerate(['z','r','g']):
        for tilepass in [1,2,3]:
            Ab = A[(A.filter == band) * (A.tilepass == tilepass)]
            H,xe,ye = np.histogram2d(Ab.ra, Ab.dec, bins=(w,h),
                                     range=((ax[1],ax[0]),(ax[2],ax[3])))
            rgb[:,:,iband] += (H.T > 0) * 255/3
    plt.clf()
    plt.imshow(rgb, extent=[ax[1],ax[0],ax[2],ax[3]], interpolation='nearest',
               origin='lower', aspect=4)
    plt.axis(ax)
    plt.title('Coverage (RGB = z/r/g)')
    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    ps.savefig()
    

    overfn = 'overlapping-ccds.fits'
    if os.path.exists(overfn):
        M = fits_table(overfn)
        M.rename('i','I')
        M.rename('j','J')
    else:
        from astrometry.libkd.spherematch import match_radec
        radius = np.hypot(2048, 4096) * 0.262/3600.
        M = fits_table()
        M.I,M.J,d = match_radec(A.ra, A.dec, A.ra, A.dec, radius, notself=True)
        M.cut(M.I < M.J)
        # ra_centers vs dra
        cosd = np.cos(np.deg2rad(A.dec_center[M.I]))
        x1 = np.cos(np.deg2rad(A.ra_center[M.I]))
        y1 = np.sin(np.deg2rad(A.ra_center[M.I]))
        x2 = np.cos(np.deg2rad(A.ra_center[M.J]))
        y2 = np.sin(np.deg2rad(A.ra_center[M.J]))
        dra = np.rad2deg(cosd * np.hypot(x1 - x2, y1 - y2))
        # widths - dra = overlap
        M.raoverlap = A.dra[M.I] + A.dra[M.J] - dra
        M.cut(M.raoverlap > 0)
            
        ddec = np.abs(A.dec_center[M.I] - A.dec_center[M.J])
        # heights - ddec = overlap
        M.decoverlap = A.ddec[M.I] + A.ddec[M.J] - ddec
        M.cut(M.decoverlap > 0)
            
        M.overlap = (M.raoverlap * M.decoverlap)
        M.dtime = np.abs(A.mjd_obs[M.I] - A.mjd_obs[M.J])
        print(len(M), 'total overlaps')
        
        # for c in A.columns():
        #     M.set('%s_1' % c, A.get(c)[M.I])
        #     M.set('%s_2' % c, A.get(c)[M.J])
        M.writeto(overfn)

    # Find images in the same band with overlapping RA,Dec ranges; plot
    # their overlap area & time lag.
    
    for band in ['g','r','z']:

        #I = np.flatnonzero((M.filter_1 == band) * (M.filter_2 == band))
        K = np.flatnonzero((A.filter[M.I] == band) * (A.filter[M.J] == band))
        MI = M[K]
        print(len(MI), band, '--', band, 'matches')

        overlaps = MI.overlap
        dtimes   = MI.dtime
        
        # plt.clf()
        # plt.plot(overlaps, dtimes, 'k.', alpha=0.1)
        # plt.xlabel('Overlap (sq deg)')
        # plt.ylabel('Delta-Times (days)')
        # plt.title('%s band repeat exposures (by CCD)' % band)
        # ps.savefig()

        plt.clf()
        plt.hist(np.log10(dtimes), bins=100, range=(-3, 3),
                 weights=overlaps)
        plt.xlabel('log Delta-Times (days)')
        plt.ylabel('Overlap area (sq deg)')
        plt.title('%s band repeat exposures (by CCD)' % band)
        yl,yh = plt.ylim()
        for d,txt in [(np.log10(1./24), '1 hour'),
                      (np.log10(5./60 * 1./24), '5 min')]:
            plt.axvline(d, color='k', linestyle='--')
            plt.text(d, (yl+yh)/2, txt, rotation=90, va='top')
        ps.savefig()

        K = np.flatnonzero((MI.dtime > 0.5) * (MI.dtime < 1.5))
        MI.cut(K)
        print(len(MI), 'in', band, 'band with delta-time 0.5 - 1.5 days')

        plt.clf()
        plt.plot(A.ra_center[MI.I], A.dec_center[MI.I], 'k.', alpha=0.1)

        for rlo,rhi in [(240, 300), (0, 150), (45, 105)]:
            ilo = np.flatnonzero(ra == rlo)[0]
            ihi = np.flatnonzero(ra == rhi)[0]
            sty = dict(lw=3, alpha=0.25)
            plt.plot([rlo,rlo], [declo[ilo],dechi[ilo]], 'k-', **sty)
            plt.plot([rhi,rhi], [declo[ihi],dechi[ihi]], 'k-', **sty)
            plt.plot(ra[ilo:ihi+1], declo[ilo:ihi+1],    'k-', **sty)
            plt.plot(ra[ilo:ihi+1], dechi[ilo:ihi+1],    'k-', **sty)
        plt.plot(ra, declo, 'k--')
        plt.plot(ra, dechi, 'k--')

        plt.axis(ax)
        plt.title('~ 1 day repeats, %s band' % band)
        plt.xlabel('RA (deg)')
        plt.ylabel('Dec (deg)')

        ps.savefig()
        
