from __future__ import print_function

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
    


    # Find images in the same band with overlapping RA,Dec ranges; plot
    # their overlap area & time lag.

    for band in ['g','r','z']:
        from astrometry.libkd.spherematch import match_radec
        
        Ab = A[A.filter == band]

        radius = np.hypot(2048, 4096) * 0.262/3600.

        I,J,d = match_radec(Ab.ra, Ab.dec, Ab.ra, Ab.dec, radius, notself=True)
        print(len(I), 'raw matches')
        K = (I < J)
        I = I[K]
        J = J[K]
        print(len(I), 'non-duplicate matches')

        # ra_center, dra
        #dra = degrees_between(Ab.ra_center[I], Ab.dec_center[I],
        #                      Ab.ra_center[J], Ab.dec_center[I])
        cosd = np.cos(np.deg2rad(Ab.dec_center[I]))
        x1 = np.cos(np.deg2rad(Ab.ra_center[I]))
        y1 = np.sin(np.deg2rad(Ab.ra_center[I]))
        x2 = np.cos(np.deg2rad(Ab.ra_center[J]))
        y2 = np.sin(np.deg2rad(Ab.ra_center[J]))
        dra = np.rad2deg(cosd * np.hypot(x1 - x2, y1 - y2))

        # widths - dra = overlap
        raoverlap = Ab.dra[I] + Ab.dra[J] - dra
        K = (raoverlap > 0)
        I = I[K]
        J = J[K]
        raoverlap = raoverlap[K]
        print(len(I), 'RA overlap')
        
        # ddec = degrees_between(Ab.ra_center[I], Ab.dec_center[I],
        #                        Ab.ra_center[I], Ab.dec_center[J])
        ddec = np.abs(Ab.dec_center[I] - Ab.dec_center[J])
                               
        # heights - ddec = overlap
        decoverlap = Ab.ddec[I] + Ab.ddec[J] - ddec
        K = (decoverlap > 0)
        I = I[K]
        J = J[K]
        raoverlap  =  raoverlap[K]
        decoverlap = decoverlap[K]
        print(len(I), 'Dec overlap')
        
        overlaps = (raoverlap * decoverlap)
        dtimes = np.abs(Ab.mjd_obs[I] - Ab.mjd_obs[J])

        #JJ = match_radec(Ab.ra, Ab.dec, Ab.ra, Ab.dec, radius, notself=True,
        #                 indexlist=True)
        # overlaps = []
        # dtimes = []
        # 
        # for i,J in enumerate(JJ):
        #     if J is None:
        #         continue
        #     J = np.array(J)
        #     J = J[J > i]
        #     if len(J) == 0:
        #         continue
        # 
        #     # ra_center, dra
        #     dra = degrees_between(Ab.ra_center[i], Ab.dec_center[i],
        #                           Ab.ra_center[J],
        #                           np.zeros(len(J)) + Ab.dec_center[i])
        #     # widths - dra = overlap
        #     raoverlap = Ab.dra[i] + Ab.dra[J] - dra
        # 
        #     ddec = degrees_between(Ab.ra_center[i], Ab.dec_center[i],
        #                            np.zeros(len(J)) + Ab.ra_center[i],
        #                            Ab.dec_center[J])
        #     # heights - ddec = overlap
        #     decoverlap = Ab.ddec[i] + Ab.ddec[J] - ddec
        # 
        #     K = np.flatnonzero((raoverlap > 0) * (decoverlap > 0))
        #     if len(K) == 0:
        #         continue
        # 
        #     overlaps.append((raoverlap * decoverlap)[K])
        #     dtimes.append(np.abs(Ab.mjd_obs[i] - Ab.mjd_obs[J])[K])
        # 
        # overlaps = np.hstack(overlaps)
        # dtimes = np.hstack(dtimes)
        print('Overlapping pairs of CCDs:', len(dtimes))

        plt.clf()
        plt.plot(overlaps, dtimes, 'k.', alpha=0.1)
        plt.xlabel('Overlap (sq deg)')
        plt.ylabel('Delta-Times (days)')
        plt.title('%s band repeat exposures (by CCD)' % band)
        ps.savefig()

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
        
