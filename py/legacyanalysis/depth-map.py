from __future__ import print_function
import pylab as plt
import numpy as np
from legacypipe.common import *
from astrometry.util.plotutils import *
from collections import Counter

def main():
    ps = PlotSequence('map')
    survey = LegacySurveyData()
    #ccds = survey.get_ccds_readonly()
    ccds = survey.get_annotated_ccds()

    # Wrap RA
    ccds.ra += (-360 * (ccds.ra > 180))

    ralo = -45.
    rahi = 45.
    declo = -5.
    dechi = 7.

    nx,ny = 800,200
    
    margin = 0.5
    
    ccds.cut((ccds.dec > declo-margin) * (ccds.dec < dechi+margin) *
             (ccds.ra  > ralo -margin) * (ccds.ra  < rahi +margin))
    print(len(ccds), 'CCDs near eBOSS-ELG box')


    I = survey.photometric_ccds(ccds)
    ccds.cut(I)
    print(len(ccds), 'photometric')
    
    I = survey.apply_blacklist(ccds)
    ccds.cut(I)
    print(len(ccds), 'after blacklist')

    # assume axis-aligned CCDs
    #ragrid = np.linspace(ralo, rahi, nx)
    #decgrid = np.linspace(declo, dechi, ny)
    rastep  = (rahi  - ralo ) / (nx-1)
    decstep = (dechi - declo) / (ny-1)

    dr = ccds.dra / np.cos(np.deg2rad(ccds.dec_center))
    ccds.dr = dr
    
    ccds.ri0 = np.round(((ccds.ra - dr) - ralo ) / rastep ).astype(int)
    ccds.ri1 = np.round(((ccds.ra + dr) - ralo ) / rastep ).astype(int)
    ccds.di0 = np.round(((ccds.dec - ccds.ddec) - declo) / decstep).astype(int)
    ccds.di1 = np.round(((ccds.dec + ccds.ddec) - declo) / decstep).astype(int)
    I = np.flatnonzero((ccds.ri1 >= 0) * (ccds.di1 >= 0) *
                       (ccds.ri0 < nx) * (ccds.di0 < ny))
    ccds.cut(I)
    print(len(ccds), 'touching plot grid')
    ccds.ri0 = np.clip(ccds.ri0, 0, nx-1)
    ccds.ri1 = np.clip(ccds.ri1, 0, nx-1)
    ccds.di0 = np.clip(ccds.di0, 0, ny-1)
    ccds.di1 = np.clip(ccds.di1, 0, ny-1)
    
    for band in 'grz':
        expmap = np.zeros((ny,nx), np.float32)
        mapn = np.zeros((ny,nx), np.float32)
        mapt = np.zeros((ny,nx), np.float32)
        I = np.flatnonzero(ccds.filter == band)
        # Split up the CCDs by exposure number and don't double-count
        expnums = np.unique(ccds.expnum[I])
        print(len(expnums), 'exposures in', band)
        for e in expnums:
            expmap[:,:] = 0
            J = np.flatnonzero(ccds.expnum[I] == e)
            J = I[J]
            for r0,r1,d0,d1,exptime in zip(ccds.ri0[J], ccds.ri1[J],
                                           ccds.di0[J], ccds.di1[J], ccds.exptime[J]):
                expmap[d0:d1+1, r0:r1+1] = 1
            mapn += expmap
            mapt += expmap * ccds.exptime[J[0]]

        plt.clf()
        dimshow(mapn, extent=(ralo, rahi, declo, dechi))
        plt.colorbar()
        plt.xlabel('RA (deg)')
        plt.ylabel('Dec (deg)')
        plt.title('Number of exposures: %s band' % band)
        ps.savefig()

        plt.clf()
        dimshow(mapt, extent=(ralo, rahi, declo, dechi))
        plt.colorbar()
        plt.xlabel('RA (deg)')
        plt.ylabel('Dec (deg)')
        plt.title('Exposure time: %s band' % band)
        ps.savefig()

        maxpixel = np.argmax(mapt)
        iy,ix = np.unravel_index(maxpixel, mapt.shape)
        print('Largest value:', mapn[iy,ix], 'exposures in', band)
        rc,dc = ralo + (ix + 0.5) * rastep, declo + (iy + 0.5) * decstep
        print('RA,Dec', rc,dc)
        J = np.flatnonzero((np.abs(ccds.ra [I] - rc) < ccds.dr[I]) *
                           (np.abs(ccds.dec[I] - dc) < ccds.ddec[I]))
        print(len(J), 'CCDs overlap that RA,Dec')
        J = I[J]
        print('Prop IDs:', Counter(ccds.propid[J]).most_common())
        print('OBJECTs:', Counter(ccds.object[J]).most_common())
        
        

if __name__ == '__main__':
    main()
    
