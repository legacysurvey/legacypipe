import numpy as np

import logging
logger = logging.getLogger('legacypipe.blobmask')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

def stage_blobmask(targetwcs=None,
                   W=None,H=None,
                   bands=None, ps=None, tims=None,
                   plots=False,
                   brickname=None,
                   version_header=None,
                   mp=None, nsigma=None,
                   survey=None, brick=None,
                   nsatur=None,
                   record_event=None,
                   blob_dilate=None,
                   **kwargs):
    from legacypipe.detection import detection_maps
    from legacypipe.runbrick import _add_stage_version

    record_event and record_event('stage_blobmask: starting')
    _add_stage_version(version_header, 'BLOBMASK', 'blobmask')

    record_event and record_event('stage_blobmask: detection maps')
    detmaps, detivs, satmaps = detection_maps(tims, targetwcs, bands, mp,
                                              apodize=10, nsatur=nsatur)

    hot, saturated_pix = generate_blobmask(
        survey, bands, nsigma, detmaps, detivs, satmaps, blob_dilate,
        version_header, targetwcs, brickname, record_event)
    del detmaps, detivs, satmaps
    print('total_pixels', np.sum([h*w for h,w in [tim.shape for tim in tims]]))

    keys = ['hot', 'saturated_pix', 'version_header', ]
    L = locals()
    rtn = dict([(k,L[k]) for k in keys])
    return rtn

def generate_blobmask(survey, bands, nsigma, detmaps, detivs, satmaps, blob_dilate,
                      version_header, targetwcs, brickname, record_event):
    from scipy.ndimage import binary_dilation
    from legacypipe.utils import copy_header_with_wcs
    from legacypipe.detection import sed_matched_detection, merge_hot_satur
    # Expand the mask around saturated pixels to avoid generating
    # peaks at the edge of the mask.
    saturated_pix = [binary_dilation(satmap > 0, iterations=4) for satmap in satmaps]
    del satmaps
    # SED-matched detections
    record_event and record_event('stage_blobmask: SED-matched')
    debug('Running source detection at', nsigma, 'sigma')
    SEDs = survey.sed_matched_filters(bands)

    H,W = detmaps[0].shape
    hot = np.zeros((H,W), bool)

    for sedname,sed in SEDs:
        sedhot = sed_matched_detection(
            sedname, sed, detmaps, detivs, bands, None, None, None,
            nsigma=nsigma, blob_dilate=blob_dilate, hotmap_only=True)
        if sedhot is None:
            continue
        hot |= sedhot
        del sedhot
    del detmaps
    del detivs

    hot = merge_hot_satur(hot, saturated_pix)

    if True:
        # For estimating compute time
        from scipy.ndimage.measurements import label, find_objects
        blobmap,_ = label(hot)
        #print('Blobmap: min', blobmap.min(), 'max', blobmap.max())
        blobslices = find_objects(blobmap)
        maxnpix = 0
        for blobid_m1,slc in enumerate(blobslices):
            npix = np.sum(blobmap[slc] == (blobid_m1 + 1))
            maxnpix = max(maxnpix, npix)
        print('max_blob_size', maxnpix)
        print('num_blobs', len(blobslices))
        del blobmap

    hdr = copy_header_with_wcs(version_header, targetwcs)
    hdr.add_record(dict(name='IMTYPE', value='blobmask',
                        comment='LegacySurveys image type'))
    with survey.write_output('blobmask', brick=brickname,
                             shape=hot.shape) as out:
        out.fits.write(hot.astype(np.uint8), header=hdr)

    return hot, saturated_pix
