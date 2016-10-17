from __future__ import print_function

import numpy as np

from astrometry.util.util import Tan
from astrometry.util.fits import fits_table

from tractor import PointSource, getParamTypeTree, RaDecPos
from tractor.galaxy import ExpGalaxy, DevGalaxy, FixedCompositeGalaxy
from tractor.ellipses import EllipseESoft, EllipseE

from legacypipe.survey import SimpleGalaxy

# FITS catalogs
fits_typemap = { PointSource: 'PSF', ExpGalaxy: 'EXP', DevGalaxy: 'DEV',
                 FixedCompositeGalaxy: 'COMP',
                 SimpleGalaxy: 'SIMP',
                 type(None): 'NONE' }

fits_short_typemap = { PointSource: 'S', ExpGalaxy: 'E', DevGalaxy: 'D',
                       FixedCompositeGalaxy: 'C',
                       SimpleGalaxy: 'G' }

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
    

def prepare_fits_catalog(cat, invvars, T, hdr, filts, fs, allbands = 'ugrizY',
                         prefix='', save_invvars=True):

    if T is None:
        T = fits_table()
    if hdr is None:
        import fitsio
        hdr = fitsio.FITSHDR()

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

    decam_flux = np.zeros((len(cat), len(allbands)), np.float32)
    decam_flux_ivar = np.zeros((len(cat), len(allbands)), np.float32)

    for filt in filts:
        flux = np.array([src is not None and
                         sum(b.getFlux(filt) for b in src.getBrightnesses())
                         for src in cat])

        if invvars is not None:
            # Oh my, this is tricky... set parameter values to the variance
            # vector so that we can read off the parameter variances via the
            # python object apis.
            cat.setParams(invvars)
            flux_iv = np.array([src is not None and
                                sum(b.getFlux(filt) for b in src.getBrightnesses())
                                for src in cat])
            cat.setParams(params0)
        else:
            flux_iv = np.zeros_like(flux)

        i = allbands.index(filt)
        decam_flux[:,i] = flux.astype(np.float32)
        decam_flux_ivar[:,i] = flux_iv.astype(np.float32)

    T.set('%sdecam_flux' % prefix, decam_flux)
    if save_invvars:
        T.set('%sdecam_flux_ivar' % prefix, decam_flux_ivar)

    if fs is not None:
        fskeys = ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix']
        for k in fskeys:
            x = getattr(fs, k)
            x = np.array(x).astype(np.float32)
            T.set('%sdecam_%s_%s' % (prefix, tim.filter, k), x.astype(np.float32))

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
    
    return T, hdr

# We'll want to compute errors in our native representation, so have a
# FITS output routine that can convert those into output format.

def _get_tractor_fits_values(T, cat, pat):
    typearray = np.array([fits_typemap[type(src)] for src in cat])
    # If there are no "COMP" sources, the type will be 'S3' rather than 'S4'...
    typearray = typearray.astype('S4')
    T.set(pat % 'type', typearray)

    T.set(pat % 'ra',  np.array([src is not None and 
                                 src.getPosition().ra  for src in cat]))
    T.set(pat % 'dec', np.array([src is not None and
                                 src.getPosition().dec for src in cat]))

    shapeExp = np.zeros((len(T), 3))
    shapeDev = np.zeros((len(T), 3))
    fracDev  = np.zeros(len(T))

    for i,src in enumerate(cat):
        if isinstance(src, ExpGalaxy):
            shapeExp[i,:] = src.shape.getAllParams()
        elif isinstance(src, DevGalaxy):
            shapeDev[i,:] = src.shape.getAllParams()
            fracDev[i] = 1.
        elif isinstance(src, FixedCompositeGalaxy):
            shapeExp[i,:] = src.shapeExp.getAllParams()
            shapeDev[i,:] = src.shapeDev.getAllParams()
            fracDev[i] = src.fracDev.getValue()

    T.set(pat % 'shapeExp', shapeExp.astype(np.float32))
    T.set(pat % 'shapeDev', shapeDev.astype(np.float32))
    T.set(pat % 'fracDev',   fracDev.astype(np.float32))
    return




def read_fits_catalog(T, hdr=None, invvars=False, bands='grz',
                      allbands = 'ugrizY', ellipseClass=EllipseE,
                      unpackShape=True):
    from tractor import NanoMaggies
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
    if hdr is None:
        hdr = T._header
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

        flux = np.atleast_1d(t.decam_flux)
        assert(np.all(np.isfinite(flux[ibands])))
        br = NanoMaggies(order=bands,
                         **dict(zip(bands, flux[ibands])))
        params = [pos, br]
        if invvars:
            # ASSUME & hard-code that the position and brightness are
            # the first params
            fluxiv = np.atleast_1d(t.decam_flux_ivar)
            ivs.extend([t.ra_ivar, t.dec_ivar] +
                       list(t.decam_flux_iv[ibands]))
            
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
