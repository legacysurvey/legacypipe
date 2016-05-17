from __future__ import print_function
import numpy as np
import fitsio
from legacypipe.cpimage import CP_DQ_BITS

def make_coadds(tims, bands, targetwcs,
            mods=None, xy=None, apertures=None, apxy=None,
            ngood=False, detmaps=False, psfsize=False,
            callback=None, callback_args=[],
            plots=False, ps=None,
            lanczos=True):
    from astrometry.util.resample import resample_with_wcs, OverlapError

    class Duck(object):
        pass
    C = Duck()

    W = int(targetwcs.get_width())
    H = int(targetwcs.get_height())

    # always, for patching SATUR, etc pixels?
    unweighted=True

    if not xy:
        psfsize = False
    
    C.coimgs = []
    if detmaps:
        C.galdetivs = []
        C.detivs = []
    if mods:
        C.comods = []
        C.coresids = []
        
    if apertures is not None:
        unweighted = True
        C.AP = fits_table()

    if xy:
        ix,iy = xy
        C.T = fits_table()
        C.T.nobs    = np.zeros((len(ix), len(bands)), np.uint8)
        C.T.anymask = np.zeros((len(ix), len(bands)), np.int16)
        C.T.allmask = np.zeros((len(ix), len(bands)), np.int16)
        if psfsize:
            C.T.psfsize = np.zeros((len(ix), len(bands)), np.float32)
        if detmaps:
            C.T.depth    = np.zeros((len(ix), len(bands)), np.float32)
            C.T.galdepth = np.zeros((len(ix), len(bands)), np.float32)

    if lanczos:
        print('Doing Lanczos resampling')

    tinyw = 1e-30
    for iband,band in enumerate(bands):
        print('Computing coadd for band', band)

        # coadded weight map (moo)
        cow    = np.zeros((H,W), np.float32)
        # coadded weighted image map
        cowimg = np.zeros((H,W), np.float32)

        kwargs = dict(cowimg=cowimg, cow=cow)

        if detmaps:
            # detection map inverse-variance (depth map)
            detiv = np.zeros((H,W), np.float32)
            C.detivs.append(detiv)
            kwargs.update(detiv=detiv)
            # galaxy detection map inverse-variance (galdepth map)
            galdetiv = np.zeros((H,W), np.float32)
            C.galdetivs.append(galdetiv)
            kwargs.update(galdetiv=galdetiv)

        if mods:
            # model image
            cowmod = np.zeros((H,W), np.float32)
            # chi-squared image
            cochi2 = np.zeros((H,W), np.float32)
            kwargs.update(cowmod=cowmod, cochi2=cochi2)

        if unweighted:
            # unweighted image
            coimg  = np.zeros((H,W), np.float32)
            if mods:
                # unweighted model
                comod  = np.zeros((H,W), np.float32)
            # number of exposures
            con    = np.zeros((H,W), np.uint8)
            # inverse-variance
            coiv   = np.zeros((H,W), np.float32)
            kwargs.update(coimg=coimg, coiv=coiv)

        # Note that we have 'congood' as well as 'nobs':
        # * 'congood' is used for the 'nexp' *image*.
        # * 'nobs' is used for the per-source measurements
        #
        # (you want to know the number of observations within the
        # source footprint, not just the peak pixel which may be
        # saturated, etc.)

        if ngood:
            congood = np.zeros((H,W), np.uint8)
            kwargs.update(congood=congood)

        if xy:
            # These match the type of the "DQ" images.
            # "any" mask
            ormask  = np.zeros((H,W), np.int16)
            # "all" mask
            andmask = np.empty((H,W), np.int16)
            allbits = reduce(np.bitwise_or, CP_DQ_BITS.values())
            andmask[:,:] = allbits
            # number of observations
            nobs  = np.zeros((H,W), np.uint8)
            kwargs.update(ormask=ormask, andmask=andmask, nobs=nobs)

        if psfsize:
            psfsizemap = np.zeros((H,W), np.float32)

        for itim,tim in enumerate(tims):
            if tim.band != band:
                continue

            if lanczos:
                from astrometry.util.miscutils import patch_image
                patched = tim.getImage().copy()
                okpix = (tim.getInvError() > 0)
                patch_image(patched, okpix)
                del okpix
                imgs = [patched]
                if mods:
                    imgs.append(mods[itim])
            else:
                imgs = []

            try:
                Yo,Xo,Yi,Xi,rimgs = resample_with_wcs(
                    targetwcs, tim.subwcs, imgs, 3)
            except OverlapError:
                continue
            if len(Yo) == 0:
                continue

            if lanczos:
                im = rimgs[0]
                if mods:
                    mo = rimgs[1]
                del patched,imgs,rimgs
            else:
                im = tim.getImage ()[Yi,Xi]
                if mods:
                    mo = mods[itim][Yi,Xi]

            iv = tim.getInvvar()[Yi,Xi]

            # invvar-weighted image
            cowimg[Yo,Xo] += iv * im
            cow   [Yo,Xo] += iv

            if unweighted:
                if tim.dq is None:
                    goodpix = 1
                else:
                    dq = tim.dq[Yi,Xi]
                    # include BLEED, SATUR, INTERP pixels if no other
                    # pixels exists (do this by eliminating all other CP
                    # flags)
                    badbits = 0
                    for bitname in ['badpix', 'cr', 'trans', 'edge', 'edge2']:
                        badbits |= CP_DQ_BITS[bitname]
                    goodpix = ((dq & badbits) == 0)
                    del dq
                    
                coimg[Yo,Xo] += goodpix * im
                con  [Yo,Xo] += goodpix
                coiv [Yo,Xo] += goodpix * 1./tim.sig1**2  # ...ish

                
            if xy:
                if tim.dq is not None:
                    dq = tim.dq[Yi,Xi]
                    ormask [Yo,Xo] |= dq
                    andmask[Yo,Xo] &= dq
                    del dq
                # raw exposure count
                nobs[Yo,Xo] += 1

            if psfsize:
                # psfnorm is in units of 1/pixels.
                # (eg, psfnorm for a gaussian is ~ 1/psf_sigma)
                # Neff is in pixels**2
                neff = 1./tim.psfnorm**2
                # Narcsec is in arcsec**2
                narcsec = neff * tim.wcs.pixel_scale()**2
                psfsizemap[Yo,Xo] += iv * (1. / narcsec)
                
            if detmaps:
                # point-source depth
                detsig1 = tim.sig1 / tim.psfnorm
                detiv[Yo,Xo] += (iv > 0) * (1. / detsig1**2)

                # Galaxy detection map
                gdetsig1 = tim.sig1 / tim.galnorm
                galdetiv[Yo,Xo] += (iv > 0) * (1. / gdetsig1**2)

            if ngood:
                congood[Yo,Xo] += (iv > 0)

            if mods:
                # straight-up
                comod[Yo,Xo] += goodpix * mo
                # invvar-weighted
                cowmod[Yo,Xo] += iv * mo
                # chi-squared
                cochi2[Yo,Xo] += iv * (im - mo)**2
                del mo
                del goodpix

            del Yo,Xo,Yi,Xi,im,iv
            # END of loop over tims

        # Per-band:
        
        cowimg /= np.maximum(cow, tinyw)
        C.coimgs.append(cowimg)
        if mods:
            cowmod  /= np.maximum(cow, tinyw)
            C.comods.append(cowmod)
            coresid = cowimg - cowmod
            coresid[cow == 0] = 0.
            C.coresids.append(coresid)

        if unweighted:
            coimg  /= np.maximum(con, 1)
            del con
            cowimg[cow == 0] = coimg[cow == 0]
            if mods:
                cowmod[cow == 0] = comod[cow == 0]

        if xy:
            C.T.nobs [:,iband] = nobs[iy,ix]
            C.T.anymask[:,iband] =  ormask [iy,ix]
            C.T.allmask[:,iband] =  andmask[iy,ix]
            # unless there were no images there...
            C.T.allmask[nobs[iy,ix] == 0, iband] = 0

            if detmaps:
                C.T.depth   [:,iband] =    detiv[iy, ix]
                C.T.galdepth[:,iband] = galdetiv[iy, ix]

        if psfsize:
            wt = cow[iy,ix]
            # psfsizemap is in units of iv * (1 / arcsec**2)
            sz = psfsizemap[iy,ix]
            sz /= np.maximum(wt, tinyw)
            sz[wt == 0] = 0.
            # Back to units of linear arcsec.
            sz = 1. / np.sqrt(sz)
            sz[wt == 0] = 0.
            # Correction factor to get back to equivalent of Gaussian sigma
            sz /= (2. * np.sqrt(np.pi))
            # Conversion factor to FWHM (2.35)
            sz *= 2. * np.sqrt(2. * np.log(2.))
            C.T.psfsize[:,iband] = sz
            del psfsizemap

        if apertures is not None:
            import photutils

            # Aperture photometry, using the unweighted "coimg" and
            # "coiv" arrays.
            with np.errstate(divide='ignore'):
                imsigma = 1.0/np.sqrt(coiv)
                imsigma[coiv == 0] = 0

            apimg = []
            apimgerr = []
            if mods:
                apres = []

            for rad in apertures:
                aper = photutils.CircularAperture(apxy, rad)
                p = photutils.aperture_photometry(coimg, aper, error=imsigma)
                apimg.append(p.field('aperture_sum'))
                apimgerr.append(p.field('aperture_sum_err'))
                if mods:
                    p = photutils.aperture_photometry(coresid, aper)
                    apres.append(p.field('aperture_sum'))
            ap = np.vstack(apimg).T
            ap[np.logical_not(np.isfinite(ap))] = 0.
            C.AP.set('apflux_img_%s' % band, ap)
            ap = 1./(np.vstack(apimgerr).T)**2
            ap[np.logical_not(np.isfinite(ap))] = 0.
            C.AP.set('apflux_img_ivar_%s' % band, ap)
            if mods:
                ap = np.vstack(apres).T
                ap[np.logical_not(np.isfinite(ap))] = 0.
                C.AP.set('apflux_resid_%s' % band, ap)
                del apres
            del apimg,apimgerr,ap

        if callback is not None:
            callback(band, *callback_args, **kwargs)
        # END of loop over bands

    return C


def write_coadd_images(band,
                       survey, brickname, version_header, tims, targetwcs,
                       cowimg=None, cow=None, cowmod=None, cochi2=None,
                       detiv=None, galdetiv=None, congood=None, **kwargs):

    # copy version_header before modifying...
    hdr = fitsio.FITSHDR()
    for r in version_header.records():
        hdr.add_record(r)
    # Grab these keywords from all input files for this band...
    keys = ['TELESCOP','OBSERVAT','OBS-LAT','OBS-LONG','OBS-ELEV',
            'INSTRUME','FILTER']
    vals = set()
    for tim in tims:
        if tim.band != band:
            continue
        v = []
        for key in keys:
            v.append(tim.primhdr.get(key,''))
        vals.add(tuple(v))
    for i,v in enumerate(vals):
        for ik,key in enumerate(keys):
            if i == 0:
                kk = key
            else:
                kk = key[:7] + '%i'%i
            hdr.add_record(dict(name=kk, value=v[ik]))
    hdr.add_record(dict(name='FILTERX', value=band))

    # DATE-OBS converted to TAI.
    # print('Times:', [tim.time for tim in tims if tim.band == band])
    mjds = [tim.time.toMjd() for tim in tims if tim.band == band]
    minmjd = min(mjds)
    maxmjd = max(mjds)
    #print('MJDs', mjds, 'range', minmjd, maxmjd)
    # back to date string in UTC...
    import astropy.time
    tt = [astropy.time.Time(mjd, format='mjd', scale='tai').utc.isot
          for mjd in [minmjd, maxmjd]]
    hdr.add_record(dict(
        name='DATEOBS1', value=tt[0],
        comment='DATE-OBS for the first image in the stack (UTC)'))
    hdr.add_record(dict(
        name='DATEOBS2', value=tt[1],
        comment='DATE-OBS for the last  image in the stack (UTC)'))

    # Plug the WCS header cards into these images
    targetwcs.add_to_header(hdr)
    hdr.delete('IMAGEW')
    hdr.delete('IMAGEH')
    hdr.add_record(dict(name='EQUINOX', value=2000.))

    imgs = [
        ('image', 'image',  cowimg),
        ]
    if congood is not None:
        imgs.append(
            ('nexp',   'expmap',   congood),
            )
    if cowmod is not None:
        imgs.extend([
                ('invvar',   'wtmap',    cow     ),
                ('model',    'model',    cowmod  ),
                ('chi2',     'chi2',     cochi2  ),
                ('depth',    'psfdepth', detiv   ),
                ('galdepth', 'galdepth', galdetiv),
                ])
    for name,prodtype,img in imgs:
        from legacypipe.common import MyFITSHDR
        hdr2 = MyFITSHDR()
        # Make a copy, because each image has different values for
        # these headers...
        #hdr2 = fitsio.FITSHDR()
        for r in hdr.records():
            hdr2.add_record(r)
        hdr2.add_record(dict(name='IMTYPE', value=name,
                             comment='LegacySurvey image type'))
        hdr2.add_record(dict(name='PRODTYPE', value=prodtype,
                             comment='NOAO image type'))
        if name in ['image', 'model']:
            hdr2.add_record(dict(name='MAGZERO', value=22.5,
                                 comment='Magnitude zeropoint'))
            hdr2.add_record(dict(name='BUNIT', value='nanomaggy',
                                 comment='AB mag = 22.5 - 2.5*log10(nanomaggy)'))
        if name in ['invvar', 'depth']:
            hdr2.add_record(dict(name='BUNIT', value='1/nanomaggy^2',
                                 comment='Ivar of ABmag=22.5-2.5*log10(nmgy)'))

        with survey.write_output(name, brick=brickname, band=band) as out:
            fitsio.write(out.fn, img, clobber=True, header=hdr2)
            print('Wrote', out.fn)

