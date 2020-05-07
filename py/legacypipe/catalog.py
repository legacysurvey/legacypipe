from __future__ import print_function

import numpy as np

from tractor import PointSource

from tractor.galaxy import ExpGalaxy, DevGalaxy
from tractor.sersic import SersicGalaxy, SersicIndex
from tractor.ellipses import EllipseE
from legacypipe.survey import RexGalaxy, GaiaSource

fits_reverse_typemap = { 'PSF': PointSource,
                         'EXP': ExpGalaxy,
                         'DEV': DevGalaxy,
                         'SER': SersicGalaxy,
                         'REX': RexGalaxy,
                         'NUN': type(None),
                         'DUP': GaiaSource,
                         }

fits_typemap = dict([(v,k) for k,v in fits_reverse_typemap.items()])
# We only want this mapping one-way
fits_typemap[GaiaSource] = 'PSF'

def _typestring(t):
    return '%s.%s' % (t.__module__, t.__name__)

def prepare_fits_catalog(cat, invvars, T, hdr, bands, allbands=None,
                         prefix='', save_invvars=True, force_keep=None):
    if T is None:
        from astrometry.util.fits import fits_table
        T = fits_table()
    if hdr is None:
        import fitsio
        hdr = fitsio.FITSHDR()
    if allbands is None:
        allbands = bands

    params0 = cat.getParams()

    flux = np.zeros((len(cat), len(allbands)), np.float32)
    flux_ivar = np.zeros((len(cat), len(allbands)), np.float32)

    for band in bands:
        i = allbands.index(band)
        for j,src in enumerate(cat):
            if src is not None:
                flux[j,i] = sum(b.getFlux(band) for b in src.getBrightnesses())

        if invvars is None:
            continue
        # Oh my, this is tricky... set parameter values to the variance
        # vector so that we can read off the parameter variances via the
        # python object apis.
        cat.setParams(invvars)

        for j,src in enumerate(cat):
            if src is not None:
                flux_ivar[j,i] = sum(b.getFlux(band) for b in src.getBrightnesses())

        cat.setParams(params0)

    T.set('%sflux' % prefix, flux)
    if save_invvars:
        T.set('%sflux_ivar' % prefix, flux_ivar)

    _get_tractor_fits_values(T, cat, '%s%%s' % prefix)

    if save_invvars:
        if invvars is not None:
            cat.setParams(invvars)
        else:
            cat.setParams(np.zeros(cat.numberOfParams()))
        _get_tractor_fits_values(T, cat, '%s%%s_ivar' % prefix)
        # Heh, "no uncertainty here!"
        T.delete_column('%stype_ivar' % prefix)
    cat.setParams(params0)

    # mod RA
    ra = T.get('%sra' % prefix)
    ra += (ra <   0) * 360.
    ra -= (ra > 360) * 360.

    # Downconvert RA,Dec invvars to float32
    for c in ['ra','dec']:
        col = '%s%s_ivar' % (prefix, c)
        T.set(col, T.get(col).astype(np.float32))

    # Zero out unconstrained values
    flux = T.get('%s%s' % (prefix, 'flux'))
    iv = T.get('%s%s' % (prefix, 'flux_ivar'))
    if force_keep is not None:
        flux[(iv == 0) * np.logical_not(force_keep[:,np.newaxis])] = 0.
    else:
        flux[iv == 0] = 0.

    return T, hdr

def _get_tractor_fits_values(T, cat, pat):
    typearray = np.array([fits_typemap[type(src)] for src in cat])
    typearray = typearray.astype('S3')
    T.set(pat % 'type', typearray)

    ra,dec = [],[]
    for src in cat:
        if src is None:
            ra.append(0.)
            dec.append(0.)
        else:
            pos = src.getPosition()
            ra.append(pos.ra)
            dec.append(pos.dec)
    T.set(pat % 'ra',  np.array(ra))
    T.set(pat % 'dec', np.array(dec))

    shape = np.zeros((len(T), 3), np.float32)
    sersic = np.zeros(len(T), np.float32)
    for i,src in enumerate(cat):
        # Grab elliptical shapes
        if isinstance(src, RexGalaxy):
            shape[i,0] = src.shape.getAllParams()[0]
        elif isinstance(src, (ExpGalaxy, DevGalaxy, SersicGalaxy)):
            shape[i,:] = src.shape.getAllParams()
        # Grab Sersic index
        if isinstance(src, SersicGalaxy):
            sersic[i] = src.sersicindex.getValue()
    T.set(pat % 'sersic',  sersic)
    T.set(pat % 'shape_r',  shape[:,0])
    T.set(pat % 'shape_e1', shape[:,1])
    T.set(pat % 'shape_e2', shape[:,2])

def read_fits_catalog(T, hdr=None, invvars=False, bands='grz', allbands=None,
                      ellipseClass=EllipseE, sersicIndexClass=SersicIndex):
    '''
    Return list of tractor Sources.

    If invvars=True, return sources,invvars
    where invvars is a list matching sources.getParams()

    If *ellipseClass* is set, assume that type for galaxy shapes; if None,
    read the type from the header.
    '''
    from tractor import NanoMaggies, RaDecPos
    if hdr is None:
        hdr = T._header
    if allbands is None:
        allbands = bands
    rev_typemap = fits_reverse_typemap

    ivs = []
    cat = []
    for t in T:
        typestr = t.type.strip()
        clazz = rev_typemap[typestr]
        assert(np.isfinite(t.ra))
        assert(np.isfinite(t.dec))
        pos = RaDecPos(t.ra, t.dec)

        fluxes = {}
        for b in bands:
            fluxes[b] = t.get('flux_' + b)
            assert(np.all(np.isfinite(fluxes[b])))
        br = NanoMaggies(order=bands, **fluxes)

        params = [pos, br]
        if invvars:
            fluxivs = []
            for b in bands:
                fluxivs.append(t.get('flux_ivar_' + b))
            ivs.extend([t.ra_ivar, t.dec_ivar] + fluxivs)

        if issubclass(clazz, (DevGalaxy, ExpGalaxy, SersicGalaxy)):
            assert(np.isfinite(t.shape_r))
            assert(np.isfinite(t.shape_e1))
            assert(np.isfinite(t.shape_e2))
            ell = ellipseClass(t.shape_r, t.shape_e1, t.shape_e2)
            params.append(ell)
            if invvars:
                ivs.extend([t.shape_r_ivar, t.shape_e1_ivar, t.shape_e2_ivar])
        elif issubclass(clazz, PointSource):
            pass
        else:
            raise RuntimeError('Unknown class %s' % str(clazz))

        if issubclass(clazz, SersicGalaxy):
            assert(np.isfinite(t.sersic))
            si = sersicIndexClass(t.sersic)
            params.append(si)
            if invvars:
                ivs.append(t.sersic_ivar)

        src = clazz(*params)
        cat.append(src)

    if invvars:
        ivs = np.array(ivs)
        ivs[np.logical_not(np.isfinite(ivs))] = 0
        return cat, ivs
    return cat
