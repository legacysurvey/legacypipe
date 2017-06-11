from __future__ import print_function

from legacypipe.runbrick import *
from legacypipe.runbrick import _depth_histogram
from legacypipe.coadds import write_coadd_images, _resample_one
from collections import OrderedDict

def stage_galdepths(survey=None, targetwcs=None, bands=None, tims=None,
                       brickname=None, version_header=None,
                       plots=False, ps=None, coadd_bw=False, W=None, H=None,
                       brick=None, blobs=None, lanczos=True, ccds=None,
                       mp=None,
                       **kwargs):
    with survey.write_output('ccds-table', brick=brickname) as out:
        ccds.writeto(None, fits_object=out.fits, primheader=version_header)

    detmaps = True
    callback=write_coadd_images
    callback_args=(survey, brickname, version_header, tims, targetwcs)
    mods = None
    
    # This is an excerpt from coadds.py
    class Duck(object):
        pass
    C = Duck()

    W = int(targetwcs.get_width())
    H = int(targetwcs.get_height())

    if detmaps:
        C.galdetivs = []
        C.detivs = []

    for tim in tims:
        # surface-brightness correction
        tim.sbscale = (targetwcs.pixel_scale() / tim.subwcs.pixel_scale())**2

    # We create one iterator per band to do the tim resampling.  These all run in
    # parallel when multi-processing.
    imaps = []
    for band in bands:
        args = []
        for itim,tim in enumerate(tims):
            if tim.band != band:
                continue
            if mods is None:
                mo = None
            else:
                mo = mods[itim]
            args.append((itim,tim,mo,lanczos,targetwcs))
        if mp is not None:
            imaps.append(mp.imap_unordered(_resample_one, args))
        else:
            import itertools
            imaps.append(itertools.imap(_resample_one, args))

    tinyw = 1e-30
    for iband,(band,timiter) in enumerate(zip(bands, imaps)):
        print('Computing coadd for band', band)
        if detmaps:
            # detection map inverse-variance (depth map)
            detiv = np.zeros((H,W), np.float32)
            C.detivs.append(detiv)
            kwargs.update(detiv=detiv)
            # galaxy detection map inverse-variance (galdepth map)
            galdetiv = np.zeros((H,W), np.float32)
            C.galdetivs.append(galdetiv)
            kwargs.update(galdetiv=galdetiv)
        
        for R in timiter:
            if R is None:
                continue

            itim,Yo,Xo,iv,im,mo,dq = R
            tim = tims[itim]
    
            if detmaps:
                # point-source depth
                detsig1 = tim.sig1 / tim.psfnorm
                detiv[Yo,Xo] += (iv > 0) * (1. / detsig1**2)

                # Galaxy detection map
                gdetsig1 = tim.sig1 / tim.galnorm
                galdetiv[Yo,Xo] += (iv > 0) * (1. / gdetsig1**2)
            del Yo,Xo,im,iv
            # END of loop over tims
        if callback is not None:

            # HACK -- don't write out the "depth" files
            depth = kwargs.pop('detiv')

            callback(band, *callback_args, **kwargs)
        # END of loop over bands

    
    
    D = _depth_histogram(brick, targetwcs, bands, C.detivs, C.galdetivs)
    with survey.write_output('depth-table', brick=brickname) as out:
        D.writeto(None, fits_object=out.fits)
    del D

    # Read and update the sha1sum file.
    olddir = '/global/projecta/projectdirs/cosmo/work/dr4c'
    fn = os.path.join(olddir, 'tractor', '%s' % brickname[:3],
                      'brick-%s.sha1sum' % brickname)
    print('Reading', fn)
    shasums = OrderedDict()
    for line in open(fn):
        words = line.split()
        val = words[0]
        key = words[1]
        key = key.replace('/global/cscratch1/sd/desiproc/dr4/data_release/dr4c/', '')
        # we shouldn't report absolute paths
        assert(key[0] != '/')
        shasums[key] = val
        print('  ', key, '=', val)

    # Update
    for fn,hashsum in survey.output_file_hashes.items():
        shasums[fn] = hashsum

    # produce per-brick checksum file.
    with survey.write_output('checksums', brick=brickname, hashsum=False) as out:
        f = open(out.fn, 'w')
        # Write our pre-computed hashcodes.
        for fn,hashsum in shasums.items():
            f.write('%s %s\n' % (hashsum, fn))
        f.close()


def main():
    from astrometry.util.multiproc import multiproc

    parser = get_parser()
    opt = parser.parse_args()
    if opt.brick is None and opt.radec is None:
        parser.print_help()
        return -1
    survey, kwargs = get_runbrick_kwargs(opt)
    if kwargs in [-1, 0]:
        return kwargs

    kwargs.update(bands = ['g','r','z'],
                  splinesky=True,
                  read_image_pixels=False,
                  mp=multiproc(),
                  )

    for k1,k2 in [('target_extent','zoom'),
                  ('W', 'width'),
                  ('H', 'height')]:
        if k2 in kwargs:
            kwargs[k1] = kwargs[k2]

    R = stage_tims(brickname=opt.brick, survey=survey,
                   **kwargs)

    stage_galdepths(**R)


if __name__ == '__main__':
    sys.exit(main())
