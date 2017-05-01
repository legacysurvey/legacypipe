from __future__ import print_function

import numpy as np

from astrometry.util.util import Tan
from astrometry.util.fits import fits_table

from tractor import PointSource, getParamTypeTree, RaDecPos
from tractor.galaxy import ExpGalaxy, DevGalaxy, FixedCompositeGalaxy
from tractor.ellipses import EllipseESoft, EllipseE

from legacypipe.survey import SimpleGalaxy, RexGalaxy

# FITS catalogs
fits_typemap = { PointSource: 'PSF', ExpGalaxy: 'EXP', DevGalaxy: 'DEV',
                 FixedCompositeGalaxy: 'COMP',
                 SimpleGalaxy: 'SIMP',
                 RexGalaxy: 'REX',
                 type(None): 'NONE' }

fits_short_typemap = { PointSource: 'S', ExpGalaxy: 'E', DevGalaxy: 'D',
                       FixedCompositeGalaxy: 'C',
                       SimpleGalaxy: 'G',
                       RexGalaxy: 'R' }

def _typestring(t):
    return '%s.%s' % (t.__module__, t.__name__)
    
ellipse_types = dict([(_typestring(t), t) for t in
                      [ EllipseESoft, EllipseE, ]])

def _source_param_types(src):
    def flatten_node(node):
        return reduce(lambda x,y: x+y,
                      [flatten_node(c) for c in node[1:]],
                      [node[0]])
    tree = getParamTypeTree(src)
    #print('Source param types:', tree)
    types = flatten_node(tree)
    return types
    

def prepare_fits_catalog(cat, invvars, T, hdr, filts, fs, allbands=None,
                         prefix='', save_invvars=True, unpackShape=True):
    if T is None:
        T = fits_table()
    if hdr is None:
        import fitsio
        hdr = fitsio.FITSHDR()
    if allbands is None:
        allbands = filts

    hdr.add_record(dict(name='TR_VER', value=1, comment='Tractor output format version'))

    # Find a source of each type and query its parameter names, for the header.
    # ASSUMES the catalog contains at least one object of each type
    for t,ts in fits_short_typemap.items():
        for src in cat:
            if type(src) != t:
                continue
            #print('Parameters for', t, src)
            sc = src.copy()
            sc.thawAllRecursive()
            for i,nm in enumerate(sc.getParamNames()):
                hdr.add_record(dict(name='TR_%s_P%i' % (ts, i), value=nm,
                                    comment='Tractor param name'))

            for i,t in enumerate(_source_param_types(sc)):
                t = _typestring(t)
                hdr.add_record(dict(name='TR_%s_T%i' % (ts, i),
                                    value=t, comment='Tractor param type'))
            break

    params0 = cat.getParams()

    flux = np.zeros((len(cat), len(allbands)), np.float32)
    flux_ivar = np.zeros((len(cat), len(allbands)), np.float32)

    for filt in filts:
        i = allbands.index(filt)
        for j,src in enumerate(cat):
            if src is not None:
                flux[j,i] = sum(b.getFlux(filt) for b in src.getBrightnesses())

        if invvars is None:
            continue
        # Oh my, this is tricky... set parameter values to the variance
        # vector so that we can read off the parameter variances via the
        # python object apis.
        cat.setParams(invvars)

        for j,src in enumerate(cat):
            if src is not None:
                flux_ivar[j,i] = sum(b.getFlux(filt) for b in src.getBrightnesses())

        cat.setParams(params0)

    T.set('%sflux' % prefix, flux)
    if save_invvars:
        T.set('%sflux_ivar' % prefix, flux_ivar)

    if fs is not None:
        fskeys = ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix']
        for k in fskeys:
            x = getattr(fs, k)
            x = np.array(x).astype(np.float32)
            T.set('%s%s_%s' % (prefix, tim.filter, k), x.astype(np.float32))

    _get_tractor_fits_values(T, cat, '%s%%s' % prefix, unpackShape=unpackShape)

    if save_invvars:
        if invvars is not None:
            cat.setParams(invvars)
        else:
            cat.setParams(np.zeros(cat.numberOfParams()))
        _get_tractor_fits_values(T, cat, '%s%%s_ivar' % prefix,
                                 unpackShape=unpackShape)
        # Heh, "no uncertainty here!"
        T.delete_column('%stype_ivar' % prefix)
    cat.setParams(params0)

    # mod
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
    flux[iv == 0] = 0.
    
    return T, hdr

# We'll want to compute errors in our native representation, so have a
# FITS output routine that can convert those into output format.

def _get_tractor_fits_values(T, cat, pat, unpackShape=True):
    typearray = np.array([fits_typemap[type(src)] for src in cat])
    # If there are no "COMP" sources, the type will be 'S3' rather than 'S4'...
    typearray = typearray.astype('S4')
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

    shapeExp = np.zeros((len(T), 3), np.float32)
    shapeDev = np.zeros((len(T), 3), np.float32)
    fracDev  = np.zeros(len(T), np.float32)

    for i,src in enumerate(cat):
        #print('_get_tractor_fits_values for pattern', pat, 'src', src)
        if isinstance(src, RexGalaxy):
            #print('Rex shape', src.shape, 'params', src.shape.getAllParams())
            shapeExp[i,0] = src.shape.getAllParams()[0]
        elif isinstance(src, ExpGalaxy):
            shapeExp[i,:] = src.shape.getAllParams()
        elif isinstance(src, DevGalaxy):
            shapeDev[i,:] = src.shape.getAllParams()
            fracDev[i] = 1.
        elif isinstance(src, FixedCompositeGalaxy):
            shapeExp[i,:] = src.shapeExp.getAllParams()
            shapeDev[i,:] = src.shapeDev.getAllParams()
            fracDev[i] = src.fracDev.getValue()

    T.set(pat % 'fracDev',   fracDev)

    if unpackShape:
        T.set(pat % 'shapeExp_r',  shapeExp[:,0])
        T.set(pat % 'shapeExp_e1', shapeExp[:,1])
        T.set(pat % 'shapeExp_e2', shapeExp[:,2])
        T.set(pat % 'shapeDev_r',  shapeDev[:,0])
        T.set(pat % 'shapeDev_e1', shapeDev[:,1])
        T.set(pat % 'shapeDev_e2', shapeDev[:,2])
    else:
        T.set(pat % 'shapeExp', shapeExp)
        T.set(pat % 'shapeDev', shapeDev)


def read_fits_catalog(T, hdr=None, invvars=False, bands='grz',
                      allbands=None, ellipseClass=EllipseE,
                      unpackShape=True, fluxPrefix='decam_'):
    '''
    This is currently a weird hybrid of dynamic and hard-coded.

    Return list of tractor Sources.

    If invvars=True, return sources,invvars
    where invvars is a list matching sources.getParams()

    If *ellipseClass* is set, assume that type for galaxy shapes; if None,
    read the type from the header.

    If *unpackShapes* is True and *ellipseClass* is EllipseE, read
    catalog entries "shapeexp_r", "shapeexp_e1", "shapeexp_e2" rather than
    "shapeExp", and similarly for "dev".
    '''
    from tractor import NanoMaggies
    if hdr is None:
        hdr = T._header
    if allbands is None:
        allbands = bands    
    rev_typemap = dict([(v,k) for k,v in fits_typemap.items()])

    if unpackShape and ellipseClass != EllipseE:
        print('Not doing unpackShape because ellipseClass != EllipseE.')
        unpackShape = False
    if unpackShape:
        T.shapeexp = np.vstack((T.shapeexp_r, T.shapeexp_e1, T.shapeexp_e2)).T
        T.shapedev = np.vstack((T.shapedev_r, T.shapedev_e1, T.shapedev_e2)).T

    ivbandcols = []

    ibands = np.array([allbands.index(b) for b in bands])

    ivs = []
    cat = []
    for i,t in enumerate(T):
        clazz = rev_typemap[t.type.strip()]
        pos = RaDecPos(t.ra, t.dec)
        assert(np.isfinite(t.ra))
        assert(np.isfinite(t.dec))

        shorttype = fits_short_typemap[clazz]

        if fluxPrefix + 'flux' in t.get_columns():
            flux = np.atleast_1d(t.get(fluxPrefix + 'flux'))
            assert(np.all(np.isfinite(flux[ibands])))
            br = NanoMaggies(order=bands,
                             **dict(zip(bands, flux[ibands])))
        else:
            fluxes = {}
            for b in bands:
                fluxes[b] = t.get(fluxPrefix + 'flux_' + b)
                assert(np.all(np.isfinite(fluxes[b])))
            br = NanoMaggies(order=bands, **fluxes)
            
        params = [pos, br]
        if invvars:
            # ASSUME & hard-code that the position and brightness are
            # the first params

            if fluxPrefix + 'flux_ivar' in t.get_columns():
                fluxiv = np.atleast_1d(t.get(fluxPrefix + 'flux_ivar'))
                fluxivs = list(fluxiv[ibands])
            else:
                fluxivs = []
                for b in bands:
                    fluxivs.append(t.get(fluxPrefix + 'flux_ivar_' + b))
            ivs.extend([t.ra_ivar, t.dec_ivar] + fluxivs)

        if issubclass(clazz, (DevGalaxy, ExpGalaxy)):
            if ellipseClass is not None:
                eclazz = ellipseClass
            else:
                # hard-code knowledge that third param is the ellipse
                eclazz = hdr['TR_%s_T3' % shorttype]
                # drop any double-quoted weirdness
                eclazz = eclazz.replace('"','')
                # look up that string... to avoid eval()
                eclazz = ellipse_types[eclazz]

            if issubclass(clazz, DevGalaxy):
                assert(np.all([np.isfinite(x) for x in t.shapedev]))
                ell = eclazz(*t.shapedev)
            else:
                assert(np.all([np.isfinite(x) for x in t.shapeexp]))
                ell = eclazz(*t.shapeexp)
            params.append(ell)
            if invvars:
                if issubclass(clazz, DevGalaxy):
                    ivs.extend(t.shapedev_ivar)
                else:
                    ivs.extend(t.shapeexp_ivar)
            
        elif issubclass(clazz, FixedCompositeGalaxy):
            # hard-code knowledge that params are fracDev, shapeE, shapeD
            assert(np.isfinite(t.fracdev))
            params.append(t.fracdev)
            if ellipseClass is not None:
                expeclazz = deveclazz = ellipseClass
            else:
                expeclazz = hdr['TR_%s_T4' % shorttype]
                deveclazz = hdr['TR_%s_T5' % shorttype]
                expeclazz = expeclazz.replace('"','')
                deveclazz = deveclazz.replace('"','')
                expeclazz = ellipse_types[expeclazz]
                deveclazz = ellipse_types[deveclazz]
            assert(np.all([np.isfinite(x) for x in t.shapedev]))
            assert(np.all([np.isfinite(x) for x in t.shapeexp]))
            ee = expeclazz(*t.shapeexp)
            de = deveclazz(*t.shapedev)
            params.append(ee)
            params.append(de)

            if invvars:
                ivs.append(t.fracdev_ivar)
                ivs.extend(t.shapeexp_ivar)
                ivs.extend(t.shapedev_ivar)

        elif issubclass(clazz, PointSource):
            pass
        else:
            raise RuntimeError('Unknown class %s' % str(clazz))

        src = clazz(*params)
        cat.append(src)

    if invvars:
        ivs = np.array(ivs)
        ivs[np.logical_not(np.isfinite(ivs))] = 0
        return cat, ivs
    return cat



if __name__ == '__main__':
    T=fits_table('3524p000-0-12-n16-sdss-cat.fits')
    cat = read_fits_catalog(T, T.get_header())
    print('Read catalog:')
    for src in cat:
        print(' ', src)
