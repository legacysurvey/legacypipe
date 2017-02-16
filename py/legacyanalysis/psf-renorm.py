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

        h,w = im.get_image_shape()

        for x,y in [(0,0), (0,h), (w,0), (w,h), (w/2, h/2)]:
            psfimg = psf.getImage(x, y)
            print('PSF image at', (x,y))#, ':', psfimg)
            print('sum', psfimg.sum())

            ph,pw = psfimg.shape
            print('psf img shape', psfimg.shape)

            apertures = [1., 3., 3.75, 5., 6., 7., 7.5, 8., 9., 10., 12.5, 15.]

            #apxy = np.array([[pw/2., ph/2.]])

            for apxy in [np.array([[pw/2., ph/2.]]),
                         np.array([[pw/2, ph/2]])]:
                print('apxy', apxy)

                ap = []
                for rad in apertures:
                    aper = photutils.CircularAperture(apxy, rad)
                    p = photutils.aperture_photometry(psfimg, aper)
                    ap.append(p.field('aperture_sum'))

                ap = np.vstack(ap).T
                print('Aperture fluxes:', ap)
