from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
#matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
import pylab as plt
import numpy as np
from astrometry.util.fits import *
from astrometry.libkd.spherematch import match_radec
from astrometry.util.plotutils import *
import fitsio
from glob import glob
from collections import Counter
from tractor.brightness import NanoMaggies

def main():

    ps = PlotSequence('dr3-summary')

    #fns = glob('/project/projectdirs/cosmo/data/legacysurvey/dr3.1/sweep/3.1/sweep-*.fits')
    #fns = glob('/project/projectdirs/cosmo/data/legacysurvey/dr3.1/sweep/3.1/sweep-240p005-250p010.fits')
    fns = ['/project/projectdirs/cosmo/data/legacysurvey/dr3.1/sweep/3.1/sweep-240p005-250p010.fits',
#           '/project/projectdirs/cosmo/data/legacysurvey/dr3.1/sweep/3.1/sweep-240p010-250p015.fits',
#           '/project/projectdirs/cosmo/data/legacysurvey/dr3.1/sweep/3.1/sweep-230p005-240p010.fits',
#           '/project/projectdirs/cosmo/data/legacysurvey/dr3.1/sweep/3.1/sweep-230p010-240p015.fits'
    ]

    TT = []
    for fn in fns:
        T = fits_table(fn)
        print(len(T), 'from', fn)
        TT.append(T)
    T = merge_tables(TT)
    del TT

    print('Types:', Counter(T.type))

    T.flux_g = T.decam_flux[:,1]
    T.flux_r = T.decam_flux[:,2]
    T.flux_z = T.decam_flux[:,4]
    T.flux_ivar_g = T.decam_flux_ivar[:,1]
    T.flux_ivar_r = T.decam_flux_ivar[:,2]
    T.flux_ivar_z = T.decam_flux_ivar[:,4]
    
    T.mag_g, T.magerr_g = NanoMaggies.fluxErrorsToMagErrors(T.flux_g, T.flux_ivar_g)
    T.mag_r, T.magerr_r = NanoMaggies.fluxErrorsToMagErrors(T.flux_r, T.flux_ivar_r)
    T.mag_z, T.magerr_z = NanoMaggies.fluxErrorsToMagErrors(T.flux_z, T.flux_ivar_z)

    T.deflux_g = T.flux_g / T.decam_mw_transmission[:,1]
    T.deflux_r = T.flux_r / T.decam_mw_transmission[:,2]
    T.deflux_z = T.flux_z / T.decam_mw_transmission[:,4]
    T.deflux_ivar_g = T.flux_ivar_g * T.decam_mw_transmission[:,1]**2
    T.deflux_ivar_r = T.flux_ivar_r * T.decam_mw_transmission[:,2]**2
    T.deflux_ivar_z = T.flux_ivar_z * T.decam_mw_transmission[:,4]**2

    T.demag_g, T.demagerr_g = NanoMaggies.fluxErrorsToMagErrors(T.deflux_g, T.deflux_ivar_g)
    T.demag_r, T.demagerr_r = NanoMaggies.fluxErrorsToMagErrors(T.deflux_r, T.deflux_ivar_r)
    T.demag_z, T.demagerr_z = NanoMaggies.fluxErrorsToMagErrors(T.deflux_z, T.deflux_ivar_z)

    T.typex = np.array([t[0] for t in T.type])

    I = np.flatnonzero((T.flux_ivar_g > 0) * (T.flux_ivar_r > 0) * (T.flux_ivar_z > 0))
    print(len(I), 'with g,r,z fluxes')
    I = np.flatnonzero((T.flux_ivar_g > 0) * (T.flux_ivar_r > 0) * (T.flux_ivar_z > 0) *
                       (T.flux_g > 0) * (T.flux_r > 0) * (T.flux_z > 0))
    print(len(I), 'with positive g,r,z fluxes')
    I = np.flatnonzero((T.flux_ivar_g > 0) * (T.flux_ivar_r > 0) * (T.flux_ivar_z > 0) *
                       (T.flux_g > 0) * (T.flux_r > 0) * (T.flux_z > 0) *
                       (T.magerr_g < 0.2) * (T.magerr_r < 0.2) * (T.magerr_z < 0.2))
    print(len(I), 'with < 0.2-mag errs')
    I = np.flatnonzero((T.flux_ivar_g > 0) * (T.flux_ivar_r > 0) * (T.flux_ivar_z > 0) *
                       (T.flux_g > 0) * (T.flux_r > 0) * (T.flux_z > 0) *
                       (T.magerr_g < 0.1) * (T.magerr_r < 0.1) * (T.magerr_z < 0.1))
    print(len(I), 'with < 0.1-mag errs')

    I2 = np.flatnonzero((T.flux_ivar_g > 0) * (T.flux_ivar_r > 0) * (T.flux_ivar_z > 0) *
                        (T.flux_g > 0) * (T.flux_r > 0) * (T.flux_z > 0) *
                        (T.magerr_g < 0.05) * (T.magerr_r < 0.05) * (T.magerr_z < 0.05))
    print(len(I2), 'with < 0.05-mag errs')

    #ha = dict(nbins=200, range=((-0.5, 2.5), (-0.5, 2.5)))
    ha = dict(nbins=400, range=((-0.5, 2.5), (-0.5, 2.5)))
    plt.clf()
    #plothist(T.mag_g[I] - T.mag_r[I], T.mag_r[I] - T.mag_z[I], 200)
    #loghist(T.mag_g[I] - T.mag_r[I], T.mag_r[I] - T.mag_z[I], **ha)
    loghist(T.demag_g[I] - T.demag_r[I], T.demag_r[I] - T.demag_z[I], **ha)
    plt.xlabel('g - r (mag)')
    plt.ylabel('r - z (mag)')
    ps.savefig()

    hists = []
    hists2 = []
    for tt,name in [('P', 'PSF'), ('E', 'EXP'), ('D', 'DeV')]:
        I = np.flatnonzero((T.flux_ivar_g > 0) * (T.flux_ivar_r > 0) *
                           (T.flux_ivar_z > 0) *
                           (T.flux_g > 0) * (T.flux_r > 0) * (T.flux_z > 0) *
                           (T.magerr_g < 0.1) * (T.magerr_r < 0.1) *(T.magerr_z < 0.1) *
                           (T.typex == tt))
        print(len(I), 'with < 0.1-mag errs and type', name)

        # plt.clf()
        # H,xe,ye = loghist(T.mag_g[I] - T.mag_r[I], T.mag_r[I] - T.mag_z[I], **ha)
        # plt.xlabel('g - r (mag)')
        # plt.ylabel('r - z (mag)')
        # plt.title('Type = %s' % name)
        # ps.savefig()

        plt.clf()
        H,xe,ye = loghist(T.demag_g[I] - T.demag_r[I], T.demag_r[I] - T.demag_z[I], **ha)

        # Keeping the dereddened color-color histograms
        hists.append(H)

        plt.xlabel('g - r (mag)')
        plt.ylabel('r - z (mag)')
        plt.title('Type = %s, dereddened' % name)
        ps.savefig()

        I2 = np.flatnonzero((T.flux_ivar_g > 0) * (T.flux_ivar_r > 0) *
                            (T.flux_ivar_z > 0) *
                            (T.flux_g > 0) * (T.flux_r > 0) * (T.flux_z > 0) *
                            (T.magerr_g < 0.05) *
                            (T.magerr_r < 0.05) *
                            (T.magerr_z < 0.05) *
                            (T.typex == tt))
        print(len(I2), 'with < 0.05-mag errs and type', name)

        plt.clf()
        H,xe,ye = loghist(T.demag_g[I2] - T.demag_r[I2], T.demag_r[I2] - T.demag_z[I2], **ha)
        hists2.append(H)
        plt.xlabel('g - r (mag)')
        plt.ylabel('r - z (mag)')
        plt.title('Type = %s, dereddened, 5%% errs' % name)
        ps.savefig()

    for H in hists:
        H /= H.max()
    #for H in hists2:
    #    H /= H.max()
    #hists2 = [np.clip(H / np.percentile(H.ravel(), 99), 0, 1) for H in hists2]
    hists2 = [np.clip(H / np.percentile(H.ravel(), 99.9), 0, 1) for H in hists2]

    rgbmap = np.dstack((hists[2], hists[0], hists[1]))
    plt.clf()
    ((xlo,xhi),(ylo,yhi)) = ha['range']
    plt.imshow(rgbmap, interpolation='nearest', origin='lower', extent=[xlo,xhi,ylo,yhi])
    plt.axis([0,2,0,2])
    plt.xlabel('g - r (mag)')
    plt.ylabel('r - z (mag)')
    plt.title('Red: DeV, Blue: Exp, Green: PSF')
    ps.savefig()

    plt.clf()
    plt.imshow(np.sqrt(rgbmap), interpolation='nearest', origin='lower', extent=[xlo,xhi,ylo,yhi])
    plt.axis([0,2,0,2])
    plt.xlabel('g - r (mag)')
    plt.ylabel('r - z (mag)')
    plt.title('Red: DeV, Blue: Exp, Green: PSF')
    ps.savefig()

    # For a white-background plot, mix colors like paint
    for hr,hg,hb in [(hists[2], hists[0], hists[1]),
                     (hists2[2], hists2[0], hists2[1]),]:
        H,W = hr.shape
        for mapping in [lambda x: x]: #, lambda x: np.sqrt(x), lambda x: x**2]:
            red = np.ones((H,W,3))
            red[:,:,1] = 1 - mapping(hr)
            red[:,:,2] = 1 - mapping(hr)
            # in matplotlib, 'green'->(0, 0.5, 0)
            green = np.ones((H,W,3))
            green[:,:,0] = 1 - mapping(hg)
            green[:,:,1] = 1 - 0.5*mapping(hg)
            green[:,:,2] = 1 - mapping(hg)
            blue = np.ones((H,W,3))
            blue[:,:,0] = 1 - mapping(hb)
            blue[:,:,1] = 1 - mapping(hb)
            plt.clf()
            plt.imshow(red * green * blue, origin='lower', interpolation='nearest',
                       extent=[xlo,xhi,ylo,yhi], aspect='auto')
            plt.xlabel('g - r (mag)')
            plt.ylabel('r - z (mag)')
            plt.xticks(np.arange(0, 2.1, 0.5))
            plt.yticks(np.arange(0, 2.1, 0.5))
            plt.axis([-0.25,2,0,2])
            plt.title('Red: DeV, Blue: Exp, Green: PSF')
            ps.savefig()


    np.random.seed(42)

    T.gr = T.mag_g - T.mag_r
    T.rz = T.mag_r - T.mag_z

    for tt,name in [('P', 'PSF'), ('E', 'EXP'), ('D', 'DeV')]:
        I = np.flatnonzero((T.flux_ivar_g > 0) * (T.flux_ivar_r > 0) *
                           (T.flux_ivar_z > 0) *
                           (T.flux_g > 0) * (T.flux_r > 0) * (T.flux_z > 0) *
                           (T.magerr_g < 0.1) * (T.magerr_r < 0.1) *(T.magerr_z < 0.1) *
                           # Bright cut
                           (T.mag_r > 16.) *

                           # Contamination cut
                           (T.decam_fracflux[:,2] < 0.5) *

                           # Mag range cut
                           (T.mag_r > 19.) *
                           (T.mag_r < 20.) *
                           
                           (T.typex == tt))
        print(len(I), 'with < 0.1-mag errs and type', name)

        # plt.clf()
        # ha = dict(histtype='step', range=(16, 24), bins=100)
        # plt.hist(T.mag_g[I], color='g', **ha)
        # plt.hist(T.mag_r[I], color='r', **ha)
        # plt.hist(T.mag_z[I], color='m', **ha)
        # plt.title('Type %s' % name)
        # ps.savefig()

        J = np.random.permutation(I)

        if tt == 'D':
            J = J[T.shapedev_r[J] < 3.]

        J = J[:100]
        rzbin = np.argsort(T.rz[J])
        rzblock = (rzbin / 10).astype(int)
        K = np.lexsort((T.gr[J], rzblock))
        #print('sorted rzblock', rzblock[K])
        #print('sorted rzbin', rzbin[K])
        J = J[K]

        # plt.clf()
        # plt.hist(T.decam_fracflux[J,2], label='fracflux_r', histtype='step', bins=20)
        # plt.hist(T.decam_fracmasked[J,2], label='fracmasked_r', histtype='step', bins=20)
        # plt.legend()
        # plt.title('Type %s' % name)
        # ps.savefig()

        #print('fracflux r', T.decam_fracflux[J,2])
        #print('fracmasked r', T.decam_fracmasked[J,2])
        #print('anymasked r', T.decam_anymask[J,2])

        imgs = []
        imgrow = []
        stackrows = []
        for i in J:
            fn = 'cutouts/cutout_%.4f_%.4f.jpg' % (T.ra[i], T.dec[i])
            if not os.path.exists(fn):
                url = 'http://legacysurvey.org/viewer/jpeg-cutout/?layer=decals-dr3&pixscale=0.262&size=25&ra=%f&dec=%f' % (T.ra[i], T.dec[i])
                print('URL', url)
                cmd = 'wget -O %s.tmp "%s" && mv %s.tmp %s' % (fn, url, fn, fn)
                os.system(cmd)
            img = plt.imread(fn)
            #print('Image', img.shape)
            extra = ''
            if tt == 'D':
                extra = ', shapedev_r %.2f' % T.shapedev_r[i]
            elif tt == 'E':
                extra = ', shapeexp_r %.2f' % T.shapeexp_r[i]

            print(name, 'RA,Dec %.4f,%.4f, g/r/z %.2f, %.2f, %.2f, fracflux_r %.3f, fracmasked_r %.3f, anymasked_r %4i%s' % (
                T.ra[i], T.dec[i], T.mag_g[i], T.mag_r[i], T.mag_z[i],
                T.decam_fracflux[i,2], T.decam_fracmasked[i,2], T.decam_anymask[i,2], extra))

            imgrow.append(img)
            if len(imgrow) == 10:
                stack = np.hstack(imgrow)
                print('stacked row:', stack.shape)
                stackrows.append(stack)

                imgs.append(imgrow)
                imgrow = []


        stack = np.vstack(stackrows)
        print('Stack', stack.shape)
        plt.clf()
        plt.imshow(stack, interpolation='nearest', origin='lower')
        plt.xticks([])
        plt.yticks([])
        plt.title('Type %s' % name)
        ps.savefig()
        

    # Wrap-around at RA=300
    # rax = T.ra + (-360 * T.ra > 300.)
    # 
    # rabins = np.arange(-60., 300., 0.2)
    # decbins = np.arange(-20, 35, 0.2)
    # 
    # print('RA bins:', len(rabins))
    # print('Dec bins:', len(decbins))
    #
    # for name,cut in [(None, 'All sources'),
    #                  (T.type == 'PSF '), 'PSF'),
    #                  (T.type == 'SIMP'), 'SIMP'),
    #                  (T.type == 'DEV '), 'DEV'),
    #                  (T.type == 'EXP '), 'EXP'),
    #                  (T.type == 'COMP'), 'COMP'),
    #                  ]:
    #                  
    # H,xe,ye = np.histogram2d(T.ra, T.dec, bins=(rabins, decbins))
    # print('H shape', H.shape)
    # 
    # H = H.T
    # 
    # H /= (rabins[1:] - rabins[:-1])[np.newaxis, :]
    # H /= ((decbins[1:] - decbins[:-1]) * np.cos(np.deg2rad((decbins[1:] + decbins[:-1]) / 2.))))[:, np.newaxis]
    # 
    # plt.clf()
    # plt.imshow(H, extent=(rabins[0], rabins[-1], decbins[0], decbins[-1]), interpolation='nearest', origin='lower',
    #            vmin=0)
    # cbar = plt.colorbar()
    # cbar.set_label('Source density per square degree')
    # plt.xlim(rabins[-1], rabins[0])
    # plt.xlabel('RA (deg)')
    # plt.ylabel('Dec (deg)')
    # ps.savefig()



    '''
  1 BRICKID          J
   2 BRICKNAME        8A
   3 OBJID            J
   4 TYPE             4A
   5 RA               D
   6 RA_IVAR          E
   7 DEC              D
   8 DEC_IVAR         E
   9 DECAM_FLUX       6E
  10 DECAM_FLUX_IVAR  6E
  11 DECAM_MW_TRANSMISSION 6E
  12 DECAM_NOBS       6B
  13 DECAM_RCHI2      6E
  14 DECAM_PSFSIZE    6E
  15 DECAM_FRACFLUX   6E
  16 DECAM_FRACMASKED 6E
  17 DECAM_FRACIN     6E
  18 DECAM_DEPTH      6E
  19 DECAM_GALDEPTH   6E
  20 OUT_OF_BOUNDS    L
  21 DECAM_ANYMASK    6I
  22 DECAM_ALLMASK    6I
  23 WISE_FLUX        4E
  24 WISE_FLUX_IVAR   4E
  25 WISE_MW_TRANSMISSION 4E
  26 WISE_NOBS        4I
  27 WISE_FRACFLUX    4E
  28 WISE_RCHI2       4E
  29 DCHISQ           5E
  30 FRACDEV          E
  31 TYCHO2INBLOB     L
  32 SHAPEDEV_R       E
  33 SHAPEDEV_R_IVAR  E
  34 SHAPEDEV_E1      E
  35 SHAPEDEV_E1_IVAR E
  36 SHAPEDEV_E2      E
  37 SHAPEDEV_E2_IVAR E
  38 SHAPEEXP_R       E
  39 SHAPEEXP_R_IVAR  E
  40 SHAPEEXP_E1      E
  41 SHAPEEXP_E1_IVAR E
  42 SHAPEEXP_E2      E
  43 SHAPEEXP_E2_IVAR E
  44 EBV              E
    '''



if __name__ == '__main__':
    main()

