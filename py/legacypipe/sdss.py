from __future__ import print_function
import os
import numpy as np
from tractor import Catalog

def get_sdss_sources(bands, targetwcs, photoobjdir=None, local=True,
                     extracols=[], ellipse=None):
    '''
    Finds SDSS catalog sources within the given `targetwcs` region,
    returning FITS table and Tractor Source objects.

    Returns
    -------
    cat : Tractor Catalog object
        Tractor Source objects for the SDSS catalog entries
    objs : fits_table object
        FITS table object for the sources.  Row-by-row parallel to `cat`.
    '''
    from astrometry.sdss import DR9, band_index, AsTransWrapper
    from astrometry.sdss.fields import read_photoobjs_in_wcs
    from tractor.sdss import get_tractor_sources_dr9

    # FIXME?
    margin = 0.

    if ellipse is None:
        ellipse = EllipseESoft.fromRAbPhi

    sdss = DR9(basedir=photoobjdir)
    if local:
        local = (local and ('BOSS_PHOTOOBJ' in os.environ)
                 and ('PHOTO_RESOLVE' in os.environ))
    if local:
        sdss.useLocalTree()

    cols = ['objid', 'ra', 'dec', 'fracdev', 'objc_type',
            'theta_dev', 'theta_deverr', 'ab_dev', 'ab_deverr',
            'phi_dev_deg',
            'theta_exp', 'theta_experr', 'ab_exp', 'ab_experr',
            'phi_exp_deg',
            'resolve_status', 'nchild', 'flags', 'objc_flags',
            'run','camcol','field','id',
            'psfflux', 'psfflux_ivar',
            'cmodelflux', 'cmodelflux_ivar',
            'modelflux', 'modelflux_ivar',
            'devflux', 'expflux', 'extinction'] + extracols

    # If we have a window_flist file cut to primary objects, use that.
    # This file comes from svn+ssh://astrometry.net/svn/trunk/projects/wise-sdss-phot
    # cut-window-flist.py, and used resolve/2013-07-29 (pre-DR13) as input.
    wfn = 'window_flist-cut.fits'
    if not os.path.exists(wfn):
        # default to the usual window_flist.fits file.
        wfn = None

    objs = read_photoobjs_in_wcs(targetwcs, margin, sdss=sdss, cols=cols, wfn=wfn)
    if objs is None:
        print('No photoObjs in wcs')
        return None,None
    print('Got', len(objs), 'photoObjs')
    print('Bands', bands, '->', list(bands))

    # It can be string-valued
    objs.objid = np.array([int(x) if len(x) else 0 for x in objs.objid])
    srcs = get_tractor_sources_dr9(
        None, None, None, objs=objs, sdss=sdss, bands=list(bands),
        nanomaggies=True, fixedComposites=True, useObjcType=True,
        ellipse=ellipse)
    print('Created', len(srcs), 'Tractor sources')

    # record coordinates in target brick image
    ok,objs.tx,objs.ty = targetwcs.radec2pixelxy(objs.ra, objs.dec)
    objs.tx -= 1
    objs.ty -= 1
    W,H = targetwcs.get_width(), targetwcs.get_height()
    objs.itx = np.clip(np.round(objs.tx), 0, W-1).astype(int)
    objs.ity = np.clip(np.round(objs.ty), 0, H-1).astype(int)

    cat = Catalog(*srcs)
    return cat, objs
