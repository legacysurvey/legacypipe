from __future__ import print_function

import sys

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import pylab as plt
import numpy as np

from astrometry.util.file import trymakedirs

from tractor import *
from tractor.psfex import *
from legacypipe.common import *

def main():
    plt.figure(figsize=(4,8))
    plt.subplots_adjust(hspace=0.2, wspace=0.2)

    #expnum = 349185
    #ccdname = 'N7'

    #djs = PixelizedPsfEx(fn='psf-c4d_140818_011313_ooi_z_v1.fits',
    #                     psfexmodel=SchlegelPsfModel)

    expnum = 396086
    ccdname = 'S31'
    djs = PixelizedPsfEx(fn='decam-00396086-S31.fits',
                         psfexmodel=SchlegelPsfModel)

    print('Schlegel PSF', djs)

    stampsz = 15

    print('Plotting bases')
    djs.psfex.plot_bases(autoscale=False)
    plt.suptitle('DJS PSF basis functions')
    plt.savefig('djs-bases.png')

    djs.psfex.plot_bases(stampsize=stampsz, autoscale=False)
    plt.suptitle('DJS PSF basis functions')
    plt.savefig('djs-bases2.png')

    H,W = 4096, 2048

    nx,ny = 6,11
    yy = np.linspace(0., H, ny+1)
    xx = np.linspace(0., W, nx+1)
    # center of cells
    yy = yy[:-1] + (yy[1]-yy[0])/2.
    xx = xx[:-1] + (xx[1]-xx[0])/2.

    mx = djs.psfex.psfbases.max()
    kwa = dict(vmin=-0.1*mx, vmax=mx)

    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    print('Plotting grid')
    djs.psfex.plot_grid(xx, yy, stampsize=stampsz, **kwa)
    plt.suptitle('DJS PSF grid')
    plt.savefig('djs-grid.png')

    for i in range(djs.psfex.nbases):
        print('Plotting grid for parameter', i)
        djs.psfex.plot_grid(xx, yy, term=i, stampsize=stampsz, **kwa)
        plt.savefig('djs-term%i.png' % i)


    decals = Decals()
    ccds = decals.find_ccds(expnum=expnum,ccdname=ccdname)
    ccd = ccds[0]
    im = decals.get_image_object(ccd)
    band = ccd.filter

    im.run_calibs()

    tim = im.get_tractor_image(pixPsf=True, nanomaggies=False, subsky=False)

    psfex = tim.psf.psfex

    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    psfex.plot_bases(autoscale=False)
    plt.suptitle('PsfEx basis functions')
    plt.savefig('psfex-bases.png')

    psfex.plot_bases(stampsize=stampsz, autoscale=False)
    plt.suptitle('PsfEx basis functions')
    plt.savefig('psfex-bases2.png')

    mx = psfex.psfbases.max()
    kwa = dict(vmin=-0.1*mx, vmax=mx)

    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    print('Plotting grid')
    psfex.plot_grid(xx, yy, stampsize=stampsz, **kwa)
    plt.suptitle('PsfEx grid')
    plt.savefig('psfex-grid.png')

    for i in range(psfex.nbases):
        print('Plotting grid for parameter', i)
        psfex.plot_grid(xx, yy, term=i, stampsize=stampsz, **kwa)
        plt.savefig('psfex-term%i.png' % i)



if __name__ == '__main__':
    main()
