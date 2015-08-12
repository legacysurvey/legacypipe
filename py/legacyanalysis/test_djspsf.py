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
    plt.subplots_adjust(hspace=0, wspace=0)

    expnum = 349185
    ccdname = 'N6'

    decals = Decals()
    ccds = decals.find_ccds(expnum=expnum,ccdname=ccdname)
    ccd = ccds[0]
    im = decals.get_image_object(ccd)
    band = ccd.filter

    tim = im.get_tractor_image(pixPsf=True, nanomaggies=False, subsky=False)

    djs = PixelizedPsfEx(fn='psf-c4d_140818_011313_ooi_z_v1.fits',
                         psfexmodel=SchlegelPsfModel)
    print('Schlegel PSF', djs)

    psfex = tim.psf

    print('Plotting bases')
    djs.psfex.plot_bases(autoscale=False)
    plt.savefig('djs-bases.png')

    H,W = tim.shape

    nx,ny = 6,11
    yy = np.linspace(0., H, ny+1)
    xx = np.linspace(0., W, nx+1)
    # center of cells
    yy = yy[:-1] + (yy[1]-yy[0])/2.
    xx = xx[:-1] + (xx[1]-xx[0])/2.

    mx = djs.psfex.psfbases.max()
    kwa = dict(vmin=-0.1*mx, vmax=mx)

    print('Plotting grid')
    djs.psfex.plot_grid(xx, yy, **kwa)
    plt.suptitle('DJS PSF grid')
    plt.savefig('djs-grid.png')

    for i in range(djs.psfex.nbases):
        print('Plotting grid for parameter', i)
        djs.psfex.plot_grid(xx, yy, term=i, **kwa)
        plt.savefig('djs-term%i.png' % i)


if __name__ == '__main__':
    main()
