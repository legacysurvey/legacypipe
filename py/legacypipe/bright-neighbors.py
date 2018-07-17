from legacypipe.survey import LegacySurveyData, wcs_for_brick, MASKBITS
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
    ps = PlotSequence('bright')

    insurvey = LegacySurveyData('/global/cscratch1/sd/desiproc/dr7out', cache_dir='/global/cscratch1/sd/dstn/dr7-depthcut')
    outsurvey = LegacySurveyData('/global/cscratch1/sd/dstn/bright',
                                 output_dir='/global/cscratch1/sd/dstn/bright')

    brickname = '0277m102'
    # adjacent = '0277m105'

    mfn = insurvey.find_file('maskbits', brick=brickname)
    #tfn = insurvey.find_file('tractor', brick=brickname)

    maskbits = fitsio.read(mfn)
    #T = fits_table(tfn)

    bright = ((maskbits & 0x2) > 0)
    print(np.sum(bright > 0), 'BRIGHT pixels set')
    primary = (maskbits & 0x1 == 0)
    print(np.sum(primary), 'PRIMARY pixels set')
    edge = binary_dilation(primary, structure=np.ones((3,3), bool))
    boundary = edge * np.logical_not(primary)
    brightedge = boundary & bright

    roi = slice(0,1000),slice(0,1000)

    def showbool(X):
        d = downsample_max(X, 8)
        h,w = X.shape
        plt.imshow(d, interpolation='nearest', origin='lower', vmin=0, vmax=1, extent=[0,w,0,h], cmap='gray')

    plt.clf()
    showbool(bright)
    plt.title('bright')
    ps.savefig()

    plt.clf()
    showbool(primary)
    plt.title('PRIMARY')
    ps.savefig()

    plt.clf()
    showbool(boundary)
    plt.title('boundary')
    ps.savefig()

    plt.clf()
    showbool(brightedge)
    #plt.imshow(edge[roi], interpolation='none', origin='lower', vmin=0, vmax=1)
    plt.title('bright at edge')
    ps.savefig()

    #left = brightedge[:,:-1] & primary[:,1:]

    brick = insurvey.get_brick_by_name(brickname)
    brickwcs = wcs_for_brick(brick)

    yy,xx = np.nonzero(brightedge)
    print(len(yy), 'bright edge pixels')
    rr,dd = brickwcs.pixelxy2radec(xx+1, yy+1)
    print('RA range', rr.min(), rr.max(), 'vs brick', brick.ra1, brick.ra2)
    print('Dec range', dd.min(), dd.max(), 'vs brick', brick.dec1, brick.dec2)

    radius = np.sqrt(2.) * 0.25 * 1.01;

    neighbors = insurvey.get_bricks_near(brick.ra, brick.dec, radius)
    print(len(neighbors), 'bricks nearby')

    #plt.clf()

    for nb in neighbors:
        if nb.brickname == brickname:
            # ignore myself!
            continue
        print('Neighbor:', nb.brickname)
        br = [nb.ra1,nb.ra1,nb.ra2,nb.ra2,nb.ra1]
        bd = [nb.dec1,nb.dec2,nb.dec2,nb.dec1,nb.dec1]
        I, = np.nonzero((rr > nb.ra1) * (rr < nb.ra2) * (dd > nb.dec1) * (dd < nb.dec2))
        if len(I) == 0:
            print('No edge pixels touch')
            #plt.plot(br,bd, 'b-')
            continue
        print('Edge pixels touch!')
        #plt.plot(br,bd, 'r-', zorder=20)

        nwcs = wcs_for_brick(nb)
        ok,x,y = nwcs.radec2pixelxy(rr[I], dd[I])
        x = np.round(x).astype(int)-1
        y = np.round(y).astype(int)-1
        print('Pixel ranges X', x.min(), x.max(), 'Y', y.min(), y.max())

        bfn = insurvey.find_file('blobmap', brick=nb.brickname)
        print('Found blob map', bfn)
        blobs = fitsio.read(bfn)
        h,w = blobs.shape
        assert(np.all((x >= 0) * (x < w) * (y >= 0) * (y < h)))
        blobvals = set(blobs[y, x])
        print('Blobs touching bright pixels:', blobvals)
        blobvals.discard(-1)

        tmap = np.zeros(blobs.max()+2, bool)
        for b in blobvals:
            tmap[b+1] = True
        touching = tmap[blobs+1]

        plt.clf()
        showbool(touching)
        plt.title('Blobs touching, brick %s' % nb.brickname)
        ps.savefig()

        mfn = insurvey.find_file('maskbits', brick=nb.brickname)
        maskbits,hdr = fitsio.read(mfn, header=True)
        #phdr = fitsio.read_header(mfn, ext=0)
        maskbits |= (MASKBITS['BRIGHT'] * touching)

        plt.clf()
        showbool((maskbits & MASKBITS['BRIGHT']) > 0)
        plt.title('New maskbits map for BRIGHT, brick %s' % nb.brickname)
        ps.savefig()

        with outsurvey.write_output('maskbits', brick=nb.brickname) as out:
            out.fits.write(maskbits, hdr=hdr)
            #print('Wrote', outsurvey.find_file('maskbits', brick=nb.brickname))
        
        tfn = insurvey.find_file('tractor', brick=nb.brickname)
        phdr = fitsio.read_header(tfn, ext=0)
        T = fits_table(tfn)
        print('Read', len(T), 'sources')
        print('Bright:', Counter(T.brightstarinblob))
        h,w = touching.shape
        iby = np.clip(np.round(T.by), 0, h-1).astype(int)
        ibx = np.clip(np.round(T.bx), 0, w-1).astype(int)
        before = np.flatnonzero(T.brightstarinblob)
        T.brightstarinblob |= touching[iby, ibx]
        print('Bright:', Counter(T.brightstarinblob))

        plt.clf()
        showbool((maskbits & MASKBITS['BRIGHT']) > 0)
        ax = plt.axis()
        after = np.flatnonzero(T.brightstarinblob)
        plt.plot(T.bx[before], T.by[before], 'gx')
        plt.plot(T.bx[after ], T.by[after ], 'r.')
        plt.axis(ax)
        plt.title('sources with brightstarinblob, brick %s' % nb.brickname)
        ps.savefig()

        with outsurvey.write_output('tractor', brick=nb.brickname) as out:
            T.writeto(None, fits_object=out.fits, primheader=phdr)
            #print('Wrote', outsurvey.find_file('tractor', brick=nb.brickname))

    #plt.plot(rr, dd, 'k.', zorder=30)
    #ps.savefig()
    
if __name__ == '__main__':
    main()
