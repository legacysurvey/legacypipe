from __future__ import print_function

from astrometry.util.fits import fits_table
from astrometry.util.plotutils import *

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
    
    
