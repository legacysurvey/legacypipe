import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
from astrometry.util.fits import *
from legacypipe.survey import *
import sys

'''

This script is to fix up the DR5 annotated-ccds file.  The
"photometric" column in that file is from the original IDL-based
zeropoints, but for DR5 processing we actually used the new legacyzpts
zeropoints, so this code computes that new cut and adds it to the annotated-ccds table.

NOTE that this script should be run on a git checkout of the "dr5.0"
tag, in order to have the correct survey.photometric_ccds() function.
(this script does not exist in that code tag, so make a copy of it).

'''

def main():
    A=fits_table('/project/projectdirs/cosmo/data/legacysurvey/dr5/ccds-annotated-dr5.fits.gz')
    
    survey = LegacySurveyData('/project/projectdirs/cosmo/data/legacysurvey/dr5')
    # this will read from the kd-tree file, which is built from A.depth_cut_ok CCDs.
    C = survey.get_ccds()
    I = survey.photometric_ccds(C)
    print(len(I), 'CCDs from the kd-tree pass the new photometric cut')
    
    A.camera = np.array([c.strip() for c in A.camera])
    I = survey.photometric_ccds(A)
    print(len(I), 'CCDs from the annotated CCDs file pass the new photometric cut')
    
    A.new_photometric = np.zeros(len(A), bool)
    A.new_photometric[I] = True
    A.new_photometric *= A.depth_cut_ok * A.has_zeropoint
    print(np.sum(A.new_photometric), 'pass depth-cut, has-zeropoint and new photometric cut')
    
    A.writeto('annotated-ccds-dr5-newphot.fits')
    
    #np.sum(A.pipeline_phot * A.depth_cut_ok * A.has_zeropoint)
    
    dropped, = np.nonzero((A.depth_cut_ok) * (A.new_photometric==False))
    #len(dropped)
    
    plt.clf()
    plt.plot(A.ra, A.dec, 'k.', alpha=0.01);
    plt.plot(A.ra[dropped], A.dec[dropped], 'r.', alpha=0.01);
    plt.axis([360,0,-30,35]);
    plt.savefig('dropped-1.png')
    
    plt.clf()
    plt.plot(A.ra, A.dec, 'k.', alpha=0.01);
    plt.plot(A.ra[dropped], A.dec[dropped], 'r.');
    plt.axis([360,0,-30,35]);
    plt.savefig('dropped-2.png')






def main2():
    A = fits_table('annotated-ccds-dr5-newphot.fits')

    I = np.flatnonzero(A.new_photometric * (A.dra == 0))
    print(len(I), 'new_photometric and dra == 0')

    # symlink annotate-ccds.py to annotate_ccds.py
    from legacypipe.annotate_ccds import main
    main(outfn='update.fits', ccds=A[I])

    up = fits_table('update.fits')
    for c in up.get_columns():
        print()
        print('Column', c)
        old = A.get(c)[I]
        #print('Updating:')
        #print(old)
        #print('to:')
        #print(up.get(c))

        for o,n in zip(old, up.get(c)):
            print(' ', '*' if np.any(o != n) else ' ', o, '->', n)

        A.get(c)[I] = up.get(c)

    A.writeto('annotated-ccds-dr5-newphot.fits')




#main2()
#sys.exit(0)







Aold = fits_table('/project/projectdirs/cosmo/data/legacysurvey/dr5/ccds-annotated-dr5.fits.gz')
Anew = fits_table('annotated-ccds-dr5-newphot.fits')

Aold.cut(Aold.depth_cut_ok * Aold.has_zeropoint)

Anew.cut(Anew.new_photometric)

C = fits_table('/project/projectdirs/cosmo/data/legacysurvey/dr5/coadd/115/1159p240/legacysurvey-1159p240-ccds.fits')

#print('dra range:', Aold.dra.min(), Aold.dra.max())
#print('ddec range:', Aold.ddec.min(), Aold.ddec.max())

for (ralo,rahi,declo,dechi) in [#(170,175,15,20)]:
        (113,119,20,25)]:
    rr,dd = np.meshgrid(np.linspace(ralo, rahi, 500), np.linspace(declo,dechi, 500))


    good = np.ones(rr.shape, bool)

    for band in ['g','r','z']:
        Anew.dracosdec = Anew.dra / np.cos(np.deg2rad(Anew.dec_center))
        Ao = Anew[(Anew.ra_center + Anew.dracosdec > ralo) *
                  (Anew.ra_center - Anew.dracosdec < rahi) *
                  (Anew.dec_center + Anew.ddec > declo) *
                  (Anew.dec_center - Anew.ddec < dechi) *
                  (Anew.filter == band)]
        print(len(Ao), 'in band', band, 'and region')

        nobs = np.zeros(rr.shape, int)
        for ccd in Ao:
            nobs[(np.abs(ccd.ra_center  - rr) < ccd.dracosdec) *
                 (np.abs(ccd.dec_center - dd) < ccd.ddec)] += 1

        plt.clf()
        plt.imshow(nobs, interpolation='nearest', origin='lower', extent=[ralo,rahi,declo,dechi])
        plt.axis([rahi,ralo,declo,dechi])
        plt.colorbar()
        plt.savefig('nobs-new-%s.png' % band)

        good[nobs < 2] = False


        if band == 'z':
            Cz = C[C.filter == 'z']
            cset = set(zip(Cz.expnum, Cz.ccdname))
            aset = set(zip(Ao.expnum, Ao.ccdname))
            print('In CCDs table but not in annotated:', cset - aset)


    plt.clf()
    plt.imshow(good, interpolation='nearest', origin='lower', extent=[ralo,rahi,declo,dechi], cmap='gray')
    plt.axis([rahi,ralo,declo,dechi])
    plt.colorbar()
    plt.savefig('nobs-new-good.png')







    good = np.ones(rr.shape, bool)

    for band in ['g','r','z']:
        Aold.dracosdec = Aold.dra / np.cos(np.deg2rad(Aold.dec_center))
        Ao = Aold[(Aold.ra_center + Aold.dracosdec > ralo) *
                  (Aold.ra_center - Aold.dracosdec < rahi) *
                  (Aold.dec_center + Aold.ddec > declo) *
                  (Aold.dec_center - Aold.ddec < dechi) *
                  (Aold.filter == band)]
        print(len(Ao), 'in band', band, 'and region')

        print('dra range', Ao.dra.min(), Ao.dra.max())
        print('ddec range', Ao.ddec.min(), Ao.ddec.max())

        nobs = np.zeros(rr.shape, int)
        for ccd in Ao:
            nobs[(np.abs(ccd.ra_center  - rr) < ccd.dracosdec) *
                 (np.abs(ccd.dec_center - dd) < ccd.ddec)] += 1

        plt.clf()
        plt.imshow(nobs, interpolation='nearest', origin='lower', extent=[ralo,rahi,declo,dechi])
        plt.axis([rahi,ralo,declo,dechi])
        plt.colorbar()
        plt.savefig('nobs-old-%s.png' % band)

        good[nobs < 2] = False

    plt.clf()
    plt.imshow(good, interpolation='nearest', origin='lower', extent=[ralo,rahi,declo,dechi], cmap='gray')
    plt.axis([rahi,ralo,declo,dechi])
    plt.colorbar()
    plt.savefig('nobs-old-good.png')


