from __future__ import print_function

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import pylab as plt
import numpy as np
import fitsio
from astrometry.util.fits import fits_table
from astrometry.util.plotutils import PlotSequence
from astrometry.util.multiproc import multiproc
from legacypipe.survey import *
from legacypipe.runs import get_survey

import photutils

import sys
import os

if __name__ == '__main__':

    ps = PlotSequence('psfnorm')

    survey = get_survey('dr4v2')
    ccds = survey.get_ccds_readonly()
    print(len(ccds), 'CCDs')
    ccds = ccds[ccds.camera == '90prime']
    print(len(ccds), 'Bok CCDs')

    ## Skip a few missing ones...
    ccds = ccds[24:]


    for i,ccd in enumerate(ccds):
        print('Reading CCD', i)
        try:
            im = survey.get_image_object(ccd, makeNewWeightMap=False)
            print('Reading', im)
            psf = im.read_psf_model(0, 0, gaussPsf=False, pixPsf=True,
                                    hybridPsf=False)
            # tim = im.get_tractor_image(gaussPsf=True, splinesky=False,
            #                            subsky=False,
            #                            nanomaggies=False,
            #                            dq=False, invvar=False)
            print('PSF:', psf)
        except:
            import traceback
            print('Failed:')
            traceback.print_exc()
            continue

        tt = 'BASS expnum %i, ccd %s' % (ccd.expnum, ccd.ccdname)

        h,w = im.get_image_shape()

        plt.clf()
        
        for x,y in [(0,0), (0,h), (w,0), (w,h), (w/2, h/2)]:
            psfimg = psf.getImage(x, y)
            print('PSF image at', (x,y))#, ':', psfimg)
            print('sum', psfimg.sum())

            ph,pw = psfimg.shape
            print('psf img shape', psfimg.shape)

            apertures = [1., 3., 3.5, 3.75, 4., 6., 8., 10.]

            #apxy = np.array([[pw/2., ph/2.]])

            #for ioff,apxy in enumerate([np.array([[pw/2., ph/2.]]),
            #                            np.array([[pw/2, ph/2]])]):
            for ioff,apxy in enumerate([np.array([[pw/2, ph/2]])]):
                print('apxy', apxy)

                ap = []
                for rad in apertures:
                    aper = photutils.CircularAperture(apxy, rad)
                    p = photutils.aperture_photometry(psfimg, aper)
                    ap.append(p.field('aperture_sum'))

                ap = np.vstack(ap).T
                ap = ap[0,:]
                print('Aperture fluxes:', ap)

                #plt.plot(apertures, ap, '.-', color='br'[ioff], label='%i,%i + %.1f,%.1f' % (x, y, apxy[0,0], apxy[0,1]))
                plt.plot(apertures, ap, '.-', label='%i,%i' % (x, y))
        plt.legend(loc='lower right')
        plt.xlabel('Aperture')
        plt.ylabel('Flux of PSF model')
        plt.ylim(0.8, 1.2)
        plt.xlim(2, 6)
        plt.axhline(1.0, color='k', alpha=0.25)
        plt.axvline(3.75, color='k', alpha=0.25)
        plt.title(tt)
        ps.savefig()

        plt.ylim(0.99, 1.01)
        plt.xlim(3.6, 3.9)
        ps.savefig()


        plt.clf()
        ap0 = None
        for ibase,psfimg in enumerate(psf.psfex.psfbases):
            ph,pw = psfimg.shape
            apxy = np.array([[pw/2, ph/2]])
            ap = []
            for rad in apertures:
                aper = photutils.CircularAperture(apxy, rad)
                p = photutils.aperture_photometry(psfimg, aper)
                ap.append(p.field('aperture_sum'))
            ap = np.vstack(ap).T
            ap = ap[0,:]
            if ibase == 0:
                ap0 = ap
                continue
            plt.plot(apertures, ap, '.-', label='Basis %i' % ibase)
        ax = plt.axis()
        plt.plot(apertures, ap0 - 1., '.-', label='Basis 0, flux-1')
        plt.axis(ax)
        plt.legend(loc='lower right')
        plt.xlabel('Aperture')
        plt.ylabel('Flux of PSF basis image')
        plt.axhline(0.0, color='k', alpha=0.25)
        plt.axvline(3.75, color='k', alpha=0.25)
        plt.title(tt)
        ps.savefig()
