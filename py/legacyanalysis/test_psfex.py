from __future__ import print_function

from astrometry.util.file import trymakedirs

from tractor import *
from legacypipe.common import *

def test_psfex(expnum, ccdname, decals_out):
    decals = Decals()
    ccds = decals.find_ccds(expnum=expnum,ccdname=ccdname)
    ccd = ccds[0]
    band = ccd.filter
    im = decals.get_image_object(ccd)
    print('Read', im)

    tim = im.get_tractor_image(gaussPsf=True, nanomaggies=False, subsky=False)
    tim.wcs = NullWCS()

    H,W = im.shape

    s0 = 4.

    dsdy = 1./2000.

    x0 = W/2.
    y0 = H/2.

    ny = 21
    nx = 11

    # Lay down a grid of point sources, where the PSF is varying in width as a
    # function of Y position.
    yy = np.linspace(0., H, ny+1)
    xx = np.linspace(0., W, nx+1)
    # center of cells
    yy = yy[:-1] + (yy[1]-yy[0])/2.
    xx = xx[:-1] + (xx[1]-xx[0])/2.

    modimg = np.zeros_like(tim.getImage())

    flux = NanoMaggies.magToNanomaggies(16.)
    star = PointSource(PixPos(0.,0.), NanoMaggies(**{ band: flux }))

    for y in yy:
        for x in xx:
            print('Rendering star at', x, y)
            s = s0 + dsdy * (y - y0)
            psf = NCircularGaussianPSF([s], [1.])
            tim.psf = psf
            star.pos.x = x
            star.pos.y = y
            mod = star.getModelPatch(tim)
            mod.addTo(modimg)

    np.random.seed(42)
    noise = np.random.normal(size=modimg.shape)
    ie = tim.getInvError()
    noise /= ie
    noise[ie == 0] = 0
    modimg += noise

    #
    imagedir = os.path.join(decals_out, 'images', 'decam')
    trymakedirs(imagedir)
    imagefn = os.path.join(imagedir, os.path.basename(ccd.image_filename.strip())).replace(
        '.fits.fz', '.fits')

    primhdr = im.read_image_primary_header()
    hdr = im.read_image_header()

    import fitsio
    fitsio.write(imagefn, None, header=primhdr, clobber=True)
    fitsio.write(imagefn, modimg, header=hdr, clobber=False)
    print('Wrote', imagefn)

    dq = im.read_dq()
    dqfn = imagefn.replace('_ooi_', '_ood_')
    fitsio.write(dqfn, None, header=primhdr, clobber=True)
    fitsio.write(dqfn, dq, header=hdr, clobber=False)
    print('Wrote', dqfn)

    iv = im.read_invvar()
    wtfn = imagefn.replace('_ooi_', '_oow_')
    fitsio.write(wtfn, None, header=primhdr, clobber=True)
    fitsio.write(wtfn, iv, header=hdr, clobber=False)
    print('Wrote', wtfn)

    # cmd = 'ln -s %s %s' % (im.dqfn, dqfn)
    # print cmd
    # os.system(cmd)
    # 
    # cmd = 'ln -s %s %s' % (im.wtfn, wtfn)
    # print cmd
    # os.system(cmd)

    ccds.image_filename = np.array(['decam/' + os.path.basename(imagefn)])
    ccds.image_hdu[0] = 1
    ccdfn = os.path.join(decals_out, 'decals-ccds.fits')
    ccds.writeto(ccdfn)
    print('Wrote', ccdfn)

    cal = os.path.join(decals_out, 'calib')
    trymakedirs(cal)

    for path in ['decals-bricks.fits', 'calib/se-config']:
        pathnm = os.path.join(decals_out, path)
        if os.path.exists(pathnm):
            continue
        cmd = 'ln -s %s/%s %s' % (decals.get_decals_dir(), path, pathnm)
        print(cmd)
        os.system(cmd)

def main():
    """
    Main routine.
    """
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--expnum', type=long, default='396086', metavar='', 
                        help='exposure number')
    parser.add_argument('-c', '--ccdname', type=str, default='S31', metavar='', 
                        help='CCD name')
    parser.add_argument('-o', '--out', type=str, default='decals-test', metavar='', 
                        help='Output directory name (DECALS_DIR)')
    args = parser.parse_args()

    test_psfex(args.expnum, args.ccdname, args.out)
    
if __name__ == "__main__":
    main()
