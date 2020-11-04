import os
from legacypipe.survey import LegacySurveyData, wcs_for_brick
from legacypipe.bits import MASKBITS
import numpy as np
import fitsio
from astrometry.util.fits import fits_table
from scipy.ndimage.morphology import binary_dilation
import matplotlib.pyplot as plt
from astrometry.util.plotutils import PlotSequence
from collections import Counter

def downsample_max(X, scale):
    H,W = X.shape
    down = np.zeros(((H+scale-1)//scale, (W+scale-1)//scale), X.dtype)
    for i in range(scale):
        for j in range(scale):
            down = np.maximum(down, X[i::scale, j::scale])
    return down

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plots', action='store_true')
    parser.add_argument('--brick', help='Brick name to run')
    parser.add_argument('--input-dir', default='/global/projecta/projectdirs/cosmo/work/legacysurvey/dr7')
    #/global/cscratch1/sd/desiproc/dr7out')
    parser.add_argument('--survey-dir', default='/global/cscratch1/sd/dstn/dr7-depthcut')
    parser.add_argument('--output-dir', default='/global/cscratch1/sd/dstn/bright')
    opt = parser.parse_args()

    plots = opt.plots
    ps = PlotSequence('bright')
    brickname = opt.brick

    insurvey = LegacySurveyData(opt.input_dir, cache_dir=opt.survey_dir)
    outsurvey = LegacySurveyData(opt.output_dir, output_dir=opt.output_dir)

    bfn = insurvey.find_file('blobmap', brick=brickname)
    print('Found blob map', bfn)
    blobs = fitsio.read(bfn)
    h,w = blobs.shape

    brick = insurvey.get_brick_by_name(brickname)
    brickwcs = wcs_for_brick(brick)
    radius = np.sqrt(2.) * 0.25 * 1.01;
    neighbors = insurvey.get_bricks_near(brick.ra, brick.dec, radius)
    print(len(neighbors), 'bricks nearby')

    def showbool(X):
        d = downsample_max(X, 8)
        h,w = X.shape
        plt.imshow(d, interpolation='nearest', origin='lower', vmin=0, vmax=1, extent=[0,w,0,h], cmap='gray')

    brightblobs = set()

    for nb in neighbors:
        if nb.brickname == brickname:
            # ignore myself!
            continue
        print('Neighbor:', nb.brickname)

        mfn = insurvey.find_file('maskbits', brick=nb.brickname)
        if not os.path.exists(mfn):
            print('No maskbits file:', mfn)
            continue
        maskbits = fitsio.read(mfn)
        bright = ((maskbits & MASKBITS['BRIGHT']) > 0)
        print(np.sum(bright > 0), 'BRIGHT pixels set')
        primary = (maskbits & MASKBITS['NPRIMARY'] == 0)
        print(np.sum(primary), 'PRIMARY pixels set')
        edge = binary_dilation(primary, structure=np.ones((3,3), bool))
        edge = edge * np.logical_not(primary)
        brightedge = edge & bright

        if plots:
            plt.clf()
            showbool(bright)
            plt.title('bright: brick %s' % nb.brickname)
            ps.savefig()

            # plt.clf()
            # showbool(primary)
            # plt.title('PRIMARY, brick %s' % nb.brickname)
            # ps.savefig()
            # 
            # plt.clf()
            # showbool(edge)
            # plt.title('boundary, brick %s' % nb.brickname)
            # ps.savefig()

            plt.clf()
            showbool(brightedge)
            plt.title('bright at edge, brick %s' % nb.brickname)
            ps.savefig()

        nwcs = wcs_for_brick(nb)

        yy,xx = np.nonzero(brightedge)
        print(len(yy), 'bright edge pixels')
        if len(yy) == 0:
            continue
        rr,dd = nwcs.pixelxy2radec(xx+1, yy+1)
        print('RA range', rr.min(), rr.max(), 'vs brick', brick.ra1, brick.ra2)
        print('Dec range', dd.min(), dd.max(), 'vs brick', brick.dec1, brick.dec2)
        # Find pixels that are within this brick's unique area
        I, = np.nonzero((rr >= brick.ra1) *  (rr <= brick.ra2) *
                        (dd >= brick.dec1) * (dd <= brick.dec2))

        if plots:
            plt.clf()
            plt.plot([brick.ra1,brick.ra1,brick.ra2,brick.ra2,brick.ra1],
                     [brick.dec1,brick.dec2,brick.dec2,brick.dec1,brick.dec1],
                     'b-')
            plt.plot(rr, dd, 'k.')
            plt.plot(rr[I], dd[I], 'r.')
            plt.title('Bright pixels from %s' % nb.brickname)
            ps.savefig()

        if len(I) == 0:
            print('No edge pixels touch')
            #plt.plot(br,bd, 'b-')
            continue
        #print('Edge pixels touch!')
        #plt.plot(br,bd, 'r-', zorder=20)

        _,x,y = brickwcs.radec2pixelxy(rr[I], dd[I])
        x = np.round(x).astype(int)-1
        y = np.round(y).astype(int)-1
        print('Pixel ranges X', x.min(), x.max(), 'Y', y.min(), y.max())
        assert(np.all((x >= 0) * (x < w) * (y >= 0) * (y < h)))
        print('Adding blobs:', np.unique(blobs[y,x]))
        brightblobs.update(blobs[y, x])
        print('Blobs touching bright pixels:', brightblobs)


    print()
    brightblobs.discard(-1)
    if len(brightblobs) == 0:
        print('No neighboring bright blobs to update!')
        return
    print('Updating', len(brightblobs), 'blobs:', brightblobs)

    tmap = np.zeros(blobs.max()+2, bool)
    for b in brightblobs:
        tmap[b+1] = True
    touching = tmap[blobs+1]

    if plots:
        plt.clf()
        showbool(touching)
        plt.title('Blobs touching bright, brick %s' % brickname)
        ps.savefig()

    mfn = insurvey.find_file('maskbits', brick=brickname)
    maskbits,hdr = fitsio.read(mfn, header=True)
    updated = maskbits | (MASKBITS['BRIGHT'] * touching)
    if np.all(maskbits == updated):
        print('No bits updated!  (Bright stars were already masked)')
        return
    maskbits = updated

    if plots:
        plt.clf()
        showbool((maskbits & MASKBITS['BRIGHT']) > 0)
        plt.title('New maskbits map for BRIGHT, brick %s' % brickname)
        ps.savefig()

    with outsurvey.write_output('maskbits', brick=brickname) as out:
        out.fits.write(maskbits, hdr=hdr)

    tfn = insurvey.find_file('tractor', brick=brickname)
    phdr = fitsio.read_header(tfn, ext=0)
    hdr = fitsio.read_header(tfn, ext=1)
    T = fits_table(tfn)
    print('Read', len(T), 'sources')
    print('Bright:', Counter(T.brightstarinblob))
    iby = np.clip(np.round(T.by), 0, h-1).astype(int)
    ibx = np.clip(np.round(T.bx), 0, w-1).astype(int)
    if plots:
        before = np.flatnonzero(T.brightstarinblob)
    T.brightstarinblob |= touching[iby, ibx]
    print('Bright:', Counter(T.brightstarinblob))

    # yuck -- copy the TUNIT headers from input to output.
    units = [hdr.get('TUNIT%i'%(i+1), '') for i in range(len(T.get_columns()))]

    if plots:
        plt.clf()
        showbool((maskbits & MASKBITS['BRIGHT']) > 0)
        ax = plt.axis()
        after = np.flatnonzero(T.brightstarinblob)
        plt.plot(T.bx[before], T.by[before], 'gx')
        plt.plot(T.bx[after ], T.by[after ], 'r.')
        plt.axis(ax)
        plt.title('sources with brightstarinblob, brick %s' % brickname)
        ps.savefig()

    with outsurvey.write_output('tractor', brick=brickname) as out:
        T.writeto(None, fits_object=out.fits, primheader=phdr, header=hdr,
                  units=units)

    #plt.plot(rr, dd, 'k.', zorder=30)
    #ps.savefig()
    
if __name__ == '__main__':
    main()
