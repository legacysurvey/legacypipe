from __future__ import print_function
import numpy as np
import fitsio
from astrometry.util.fits import fits_table
from astrometry.util.resample import resample_with_wcs, OverlapError
from legacypipe.bits import DQ_BITS
from legacypipe.survey import tim_get_resamp

import logging
logger = logging.getLogger('legacypipe.coadds')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

class UnwiseCoadd(object):
    def __init__(self, targetwcs, W, H, pixscale, wpixscale):
        from legacypipe.survey import wcs_for_brick, BrickDuck

        self.wW = int(W * pixscale / wpixscale)
        self.wH = int(H * pixscale / wpixscale)
        rc,dc = targetwcs.radec_center()
        # quack
        brick = BrickDuck(rc, dc, 'quack')
        self.unwise_wcs = wcs_for_brick(brick, W=self.wW, H=self.wH,
                                        pixscale=wpixscale)
        # images
        self.unwise_co  = [np.zeros((self.wH,self.wW), np.float32)
                           for band in [1,2,3,4]]
        self.unwise_con = [np.zeros((self.wH,self.wW), np.uint16)
                           for band in [1,2,3,4]]
        # models
        self.unwise_com  = [np.zeros((self.wH,self.wW), np.float32)
                            for band in [1,2,3,4]]
        # invvars
        self.unwise_coiv  = [np.zeros((self.wH,self.wW), np.float32)
                             for band in [1,2,3,4]]

    def add(self, tile, wise_models):
        for band in [1,2,3,4]:
            if not (tile, band) in wise_models:
                debug('Tile', tile, 'band', band, '-- model not found')
                continue

            # With the move_crpix option (Aaron's updated astrometry),
            # the WCS for each band can be different, so we call resample_with_wcs
            # for each band with (potentially) slightly different WCSes.
            (mod, img, ie, _, wcs) = wise_models[(tile, band)]
            debug('WISE: resampling', wcs, 'to', self.unwise_wcs)
            try:
                Yo,Xo,Yi,Xi,resam = resample_with_wcs(self.unwise_wcs, wcs,
                                                      [img, mod], intType=np.int16)
                rimg,rmod = resam
                debug('Adding', len(Yo), 'pixels from tile', tile, 'to coadd')
                self.unwise_co [band-1][Yo,Xo] += rimg
                self.unwise_com[band-1][Yo,Xo] += rmod
                self.unwise_con[band-1][Yo,Xo] += 1
                self.unwise_coiv[band-1][Yo,Xo] += ie[Yi, Xi]**2
                debug('Band', band, ': now', np.sum(self.unwise_con[band-1]>0), 'pixels are set in image coadd')
            except OverlapError:
                debug('No overlap between WISE model tile', tile, 'and brick')

    def finish(self, survey, brickname, version_header):
        from legacypipe.survey import imsave_jpeg
        for band,co,n,com,coiv in zip([1,2,3,4],
                                      self.unwise_co,  self.unwise_con, self.unwise_com, self.unwise_coiv):
            hdr = fitsio.FITSHDR()
            for r in version_header.records():
                hdr.add_record(r)
            hdr.add_record(dict(name='TELESCOP', value='WISE'))
            hdr.add_record(dict(name='FILTER', value='W%i' % band,
                                    comment='WISE band'))
            self.unwise_wcs.add_to_header(hdr)
            hdr.delete('IMAGEW')
            hdr.delete('IMAGEH')
            hdr.add_record(dict(name='EQUINOX', value=2000.))
            hdr.add_record(dict(name='MAGZERO', value=22.5,
                                    comment='Magnitude zeropoint'))
            hdr.add_record(dict(name='MAGSYS', value='Vega',
                                    comment='This WISE image is in Vega fluxes'))
            co  /= np.maximum(n, 1)
            com /= np.maximum(n, 1)
            with survey.write_output('image', brick=brickname, band='W%i' % band,
                                     shape=co.shape) as out:
                out.fits.write(co, header=hdr)
            with survey.write_output('model', brick=brickname, band='W%i' % band,
                                     shape=com.shape) as out:
                out.fits.write(com, header=hdr)
            with survey.write_output('invvar', brick=brickname, band='W%i' % band,
                                     shape=co.shape) as out:
                out.fits.write(coiv, header=hdr)
        # W1/W2 color jpeg
        rgb = _unwise_to_rgb(self.unwise_co[:2])
        with survey.write_output('wise-jpeg', brick=brickname) as out:
            imsave_jpeg(out.fn, rgb, origin='lower')
            info('Wrote', out.fn)
        rgb = _unwise_to_rgb(self.unwise_com[:2])
        with survey.write_output('wisemodel-jpeg', brick=brickname) as out:
            imsave_jpeg(out.fn, rgb, origin='lower')
            info('Wrote', out.fn)

def _unwise_to_rgb(imgs):
    img = imgs[0]
    H,W = img.shape
    ## FIXME
    w1,w2 = imgs
    rgb = np.zeros((H, W, 3), np.uint8)
    scale1 = 50.
    scale2 = 50.
    mn,mx = -1.,100.
    arcsinh = 1.
    img1 = w1 / scale1
    img2 = w2 / scale2
    if arcsinh is not None:
        def nlmap(x):
            return np.arcsinh(x * arcsinh) / np.sqrt(arcsinh)
        mean = (img1 + img2) / 2.
        I = nlmap(mean)
        img1 = img1 / mean * I
        img2 = img2 / mean * I
        mn = nlmap(mn)
        mx = nlmap(mx)
    img1 = (img1 - mn) / (mx - mn)
    img2 = (img2 - mn) / (mx - mn)
    rgb[:,:,2] = (np.clip(img1, 0., 1.) * 255).astype(np.uint8)
    rgb[:,:,0] = (np.clip(img2, 0., 1.) * 255).astype(np.uint8)
    rgb[:,:,1] = rgb[:,:,0]/2 + rgb[:,:,2]/2
    return rgb

def make_coadds(tims, bands, targetwcs,
                mods=None, blobmods=None,
                xy=None, apertures=None, apxy=None,
                ngood=False, detmaps=False, psfsize=False, allmasks=True,
                max=False, sbscale=True,
                psf_images=False,
                callback=None, callback_args=None,
                plots=False, ps=None,
                lanczos=True, mp=None,
                satur_val=10.):
    from astrometry.util.ttime import Time
    t0 = Time()

    if callback_args is None:
        callback_args = []

    class Duck(object):
        pass
    C = Duck()

    W = int(targetwcs.get_width())
    H = int(targetwcs.get_height())

    # always, for patching SATUR, etc pixels?
    unweighted=True

    C.coimgs = []
    # the pixelwise inverse-variances (weights) of the "coimgs".
    C.cowimgs = []
    if detmaps:
        C.galdetivs = []
        C.psfdetivs = []
    if mods is not None:
        C.comods = []
        C.coresids = []
    if blobmods is not None:
        C.coblobmods = []
        C.coblobresids = []
    if apertures is not None:
        C.AP = fits_table()
    if allmasks:
        C.allmasks = []
    if max:
        C.maximgs = []
    if psf_images:
        C.psf_imgs = []

    if xy:
        ix,iy = xy
        C.T = fits_table()
        C.T.nobs    = np.zeros((len(ix), len(bands)), np.int16)
        C.T.anymask = np.zeros((len(ix), len(bands)), np.int16)
        C.T.allmask = np.zeros((len(ix), len(bands)), np.int16)
        if psfsize:
            C.T.psfsize = np.zeros((len(ix), len(bands)), np.float32)
        if detmaps:
            C.T.psfdepth = np.zeros((len(ix), len(bands)), np.float32)
            C.T.galdepth = np.zeros((len(ix), len(bands)), np.float32)

    if lanczos:
        debug('Doing Lanczos resampling')

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
            if blobmods is None:
                bmo = None
            else:
                bmo = blobmods[itim]
            args.append((itim,tim,mo,bmo,lanczos,targetwcs,sbscale))
        if mp is not None:
            imaps.append(mp.imap_unordered(_resample_one, args))
        else:
            imaps.append(map(_resample_one, args))

    # Args for aperture photometry
    apargs = []

    if xy:
        # To save the memory of 2 x float64 maps, we instead do arg min/max maps

        # append a 0 to the list of mjds so that mjds[-1] gives 0.
        mjds = np.array([tim.time.toMjd() for tim in tims] + [0])
        mjd_argmins = np.empty((H,W), np.int16)
        mjd_argmaxs = np.empty((H,W), np.int16)
        mjd_argmins[:,:] = -1
        mjd_argmaxs[:,:] = -1

    if plots:
        allresids = []

    tinyw = 1e-30
    for iband,(band,timiter) in enumerate(zip(bands, imaps)):
        debug('Computing coadd for band', band)

        # coadded weight map (moo)
        cow    = np.zeros((H,W), np.float32)
        # coadded weighted image map
        cowimg = np.zeros((H,W), np.float32)

        kwargs = dict(cowimg=cowimg, cow=cow)

        if detmaps:
            # detection map inverse-variance (depth map)
            psfdetiv = np.zeros((H,W), np.float32)
            C.psfdetivs.append(psfdetiv)
            kwargs.update(psfdetiv=psfdetiv)
            # galaxy detection map inverse-variance (galdepth map)
            galdetiv = np.zeros((H,W), np.float32)
            C.galdetivs.append(galdetiv)
            kwargs.update(galdetiv=galdetiv)

        if mods is not None:
            # model image
            cowmod = np.zeros((H,W), np.float32)
            # chi-squared image
            cochi2 = np.zeros((H,W), np.float32)
            kwargs.update(cowmod=cowmod, cochi2=cochi2)

        if blobmods is not None:
            # model image
            cowblobmod = np.zeros((H,W), np.float32)
            kwargs.update(cowblobmod=cowblobmod)

        if unweighted:
            # unweighted image
            coimg  = np.zeros((H,W), np.float32)
            if mods is not None:
                # unweighted model
                comod  = np.zeros((H,W), np.float32)
            if blobmods is not None:
                coblobmod  = np.zeros((H,W), np.float32)
            # number of exposures
            con    = np.zeros((H,W), np.int16)
            # inverse-variance
            coiv   = np.zeros((H,W), np.float32)
            kwargs.update(coimg=coimg, coiv=coiv)

        # Note that we have 'congood' as well as 'nobs':
        # * 'congood' is used for the 'nexp' *image*.
        #   It counts the number of "good" (unmasked) exposures
        # * 'nobs' is used for the per-source measurements
        #   It counts the total number of exposures, including masked pixels
        #
        # (you want to know the number of observations within the
        # source footprint, not just the peak pixel which may be
        # saturated, etc.)

        if ngood:
            congood = np.zeros((H,W), np.int16)
            kwargs.update(congood=congood)

        if xy or allmasks:
            # These match the type of the "DQ" images.
            # "any" mask
            ormask  = np.zeros((H,W), np.int16)
            # "all" mask
            andmask = np.empty((H,W), np.int16)
            from functools import reduce
            allbits = reduce(np.bitwise_or, DQ_BITS.values())
            andmask[:,:] = allbits
            kwargs.update(ormask=ormask, andmask=andmask)
        if xy:
            # number of observations
            nobs = np.zeros((H,W), np.int16)
            kwargs.update(nobs=nobs)

        if psfsize:
            psfsizemap = np.zeros((H,W), np.float32)
            # like "cow", but constant invvar per-CCD;
            # only required for psfsizemap
            flatcow = np.zeros((H,W), np.float32)
            kwargs.update(psfsize=psfsizemap)

        if max:
            maximg = np.zeros((H,W), np.float32)
            C.maximgs.append(maximg)

        if psf_images:
            psf_img = 0.

        for R in timiter:
            if R is None:
                continue
            itim,Yo,Xo,iv,im,mo,bmo,dq = R
            tim = tims[itim]

            if plots:
                _make_coadds_plots_1(im, band, mods, mo, iv, unweighted,
                                     dq, satur_val, allresids, ps, H, W,
                                     tim, Yo, Xo)
            # invvar-weighted image
            cowimg[Yo,Xo] += iv * im
            cow   [Yo,Xo] += iv

            goodpix = None
            if unweighted:
                if dq is None:
                    goodpix = 1
                else:
                    # include SATUR pixels if no other
                    # pixels exists
                    okbits = 0
                    for bitname in ['satur']:
                        okbits |= DQ_BITS[bitname]
                    brightpix = ((dq & okbits) != 0)
                    if satur_val is not None:
                        # HACK -- force SATUR pix to be bright
                        im[brightpix] = satur_val
                    # Include these pixels if none other exist??
                    for bitname in ['interp']: #, 'bleed']:
                        okbits |= DQ_BITS[bitname]
                    goodpix = ((dq & ~okbits) == 0)

                coimg[Yo,Xo] += goodpix * im
                con  [Yo,Xo] += goodpix
                coiv [Yo,Xo] += goodpix * 1./(tim.sig1 * tim.sbscale)**2  # ...ish

            if xy or allmasks:
                if dq is not None:
                    ormask [Yo,Xo] |= dq
                    andmask[Yo,Xo] &= dq
            if xy:
                # raw exposure count
                nobs[Yo,Xo] += 1
                # mjd_min/max
                update = np.logical_or(mjd_argmins[Yo,Xo] == -1,
                                       (mjd_argmins[Yo,Xo] > -1) *
                                       (mjds[itim] < mjds[mjd_argmins[Yo,Xo]]))
                mjd_argmins[Yo[update],Xo[update]] = itim
                update = np.logical_or(mjd_argmaxs[Yo,Xo] == -1,
                                       (mjd_argmaxs[Yo,Xo] > -1) *
                                       (mjds[itim] > mjds[mjd_argmaxs[Yo,Xo]]))
                mjd_argmaxs[Yo[update],Xo[update]] = itim
                del update

            if psfsize:
                # psfnorm is in units of 1/pixels.
                # (eg, psfnorm for a gaussian is 1./(2.*sqrt(pi) * psf_sigma) )
                # Neff is in pixels**2
                neff = 1./tim.psfnorm**2
                # Narcsec is in arcsec**2
                narcsec = neff * tim.wcs.pixel_scale()**2
                # Make smooth maps -- don't ignore CRs, saturated pix, etc
                iv1 = 1./tim.sig1**2
                psfsizemap[Yo,Xo] += iv1 * (1. / narcsec)
                flatcow   [Yo,Xo] += iv1

            if psf_images:
                from astrometry.util.util import lanczos3_interpolate
                h,w = tim.shape
                patch = tim.psf.getPointSourcePatch(w//2, h//2).patch
                patch /= np.sum(patch)
                # In case the tim and coadd have different pixel scales,
                # resample the PSF stamp.
                ph,pw = patch.shape
                pscale = tim.imobj.pixscale / targetwcs.pixel_scale()
                coph = int(np.ceil(ph * pscale))
                copw = int(np.ceil(pw * pscale))
                coph = 2 * (coph//2) + 1
                copw = 2 * (copw//2) + 1
                # want input image pixel coords that change by 1/pscale
                # and are centered on pw//2, ph//2
                cox = np.arange(copw) * 1./pscale
                cox += pw//2 - cox[copw//2]
                coy = np.arange(coph) * 1./pscale
                coy += ph//2 - coy[coph//2]
                fx,fy = np.meshgrid(cox,coy)
                fx = fx.ravel()
                fy = fy.ravel()
                ix = (fx + 0.5).astype(np.int32)
                iy = (fy + 0.5).astype(np.int32)
                dx = (fx - ix).astype(np.float32)
                dy = (fy - iy).astype(np.float32)
                copsf = np.zeros(coph*copw, np.float32)
                rtn = lanczos3_interpolate(ix, iy, dx, dy, [copsf], [patch])
                assert(rtn == 0)
                copsf = copsf.reshape((coph,copw))
                copsf /= copsf.sum()
                if plots:
                    _make_coadds_plots_2(patch, copsf, psf_img, tim, band, ps)

                psf_img += copsf / tim.sig1**2

            if detmaps:
                # point-source depth
                detsig1 = tim.sig1 / tim.psfnorm
                psfdetiv[Yo,Xo] += (iv > 0) * (1. / detsig1**2)
                # Galaxy detection map
                gdetsig1 = tim.sig1 / tim.galnorm
                galdetiv[Yo,Xo] += (iv > 0) * (1. / gdetsig1**2)

            if ngood:
                congood[Yo,Xo] += (iv > 0)

            if mods is not None:
                # straight-up
                comod[Yo,Xo] += goodpix * mo
                # invvar-weighted
                cowmod[Yo,Xo] += iv * mo
                # chi-squared
                cochi2[Yo,Xo] += iv * (im - mo)**2
                del mo

            if blobmods is not None:
                # straight-up
                coblobmod[Yo,Xo] += goodpix * bmo
                # invvar-weighted
                cowblobmod[Yo,Xo] += iv * bmo
                del bmo
            del goodpix

            if max:
                maximg[Yo,Xo] = np.maximum(maximg[Yo,Xo], im * (iv>0))

            del Yo,Xo,im,iv
            # END of loop over tims
        # Per-band:
        cowimg /= np.maximum(cow, tinyw)
        C.coimgs.append(cowimg)
        C.cowimgs.append(cow)
        if mods is not None:
            cowmod  /= np.maximum(cow, tinyw)
            C.comods.append(cowmod)
            coresid = cowimg - cowmod
            coresid[cow == 0] = 0.
            C.coresids.append(coresid)

        if blobmods is not None:
            cowblobmod  /= np.maximum(cow, tinyw)
            C.coblobmods.append(cowblobmod)
            coblobresid = cowimg - cowblobmod
            coblobresid[cow == 0] = 0.
            C.coblobresids.append(coblobresid)

        if allmasks:
            C.allmasks.append(andmask)

        if psf_images:
            C.psf_imgs.append(psf_img / np.sum(psf_img))

        if unweighted:
            coimg  /= np.maximum(con, 1)
            del con

            if plots:
                _make_coadds_plots_3(cowimg, cow, coimg, band, ps)

            cowimg[cow == 0] = coimg[cow == 0]
            if mods is not None:
                cowmod[cow == 0] = comod[cow == 0]
            if blobmods is not None:
                cowblobmod[cow == 0] = coblobmod[cow == 0]

        if xy:
            C.T.nobs   [:,iband] = nobs   [iy,ix]
            C.T.anymask[:,iband] = ormask [iy,ix]
            C.T.allmask[:,iband] = andmask[iy,ix]
            # unless there were no images there...
            C.T.allmask[nobs[iy,ix] == 0, iband] = 0
            if detmaps:
                C.T.psfdepth[:,iband] = psfdetiv[iy, ix]
                C.T.galdepth[:,iband] = galdetiv[iy, ix]

        if psfsize:
            # psfsizemap is accumulated in units of iv * (1 / arcsec**2)
            # take out the weighting
            psfsizemap /= np.maximum(flatcow, tinyw)
            # Correction factor to get back to equivalent of Gaussian sigma
            tosigma = 1./(2. * np.sqrt(np.pi))
            # Conversion factor to FWHM (2.35)
            tofwhm = 2. * np.sqrt(2. * np.log(2.))
            # Scale back to units of linear arcsec.
            psfsizemap[:,:] = (1. / np.sqrt(psfsizemap)) * tosigma * tofwhm
            psfsizemap[flatcow == 0] = 0.
            if xy:
                C.T.psfsize[:,iband] = psfsizemap[iy,ix]

        if apertures is not None:
            # Aperture photometry
            # photutils.aperture_photometry: mask=True means IGNORE
            mask = (cow == 0)
            with np.errstate(divide='ignore'):
                imsigma = 1.0/np.sqrt(cow)
                imsigma[mask] = 0.

            for irad,rad in enumerate(apertures):
                apargs.append((irad, band, rad, cowimg, imsigma, mask,
                               True, apxy))
                if mods is not None:
                    apargs.append((irad, band, rad, coresid, None, None,
                                   False, apxy))
                if blobmods is not None:
                    apargs.append((irad, band, rad, coblobresid, None, None,
                                   False, apxy))

        if callback is not None:
            callback(band, *callback_args, **kwargs)
        # END of loop over bands

    t2 = Time()
    debug('coadds: images:', t2-t0)

    if plots:
        _make_coadds_plots_4(allresids, mods, ps)

    if xy is not None:
        C.T.mjd_min = mjds[mjd_argmins[iy,ix]]
        C.T.mjd_max = mjds[mjd_argmaxs[iy,ix]]
        del mjd_argmins
        del mjd_argmaxs

    if apertures is not None:
        # Aperture phot, in parallel
        if mp is not None:
            apresults = mp.map(_apphot_one, apargs)
        else:
            apresults = map(_apphot_one, apargs)
        del apargs
        apresults = iter(apresults)

        for iband,band in enumerate(bands):
            apimg = []
            apimgerr = []
            apmask = []
            if mods is not None:
                apres = []
            if blobmods is not None:
                apblobres = []
            for irad,rad in enumerate(apertures):
                (airad, aband, isimg, ap_img, ap_err, ap_mask) = next(apresults)
                assert(airad == irad)
                assert(aband == band)
                assert(isimg)
                apimg.append(ap_img)
                apimgerr.append(ap_err)
                apmask.append(ap_mask)

                if mods is not None:
                    (airad, aband, isimg, ap_img, ap_err, ap_mask) = next(apresults)
                    assert(airad == irad)
                    assert(aband == band)
                    assert(not isimg)
                    apres.append(ap_img)
                    assert(ap_err is None)
                    assert(ap_mask is None)

                if blobmods is not None:
                    (airad, aband, isimg, ap_img, ap_err, ap_mask) = next(apresults)
                    assert(airad == irad)
                    assert(aband == band)
                    assert(not isimg)
                    apblobres.append(ap_img)
                    assert(ap_err is None)
                    assert(ap_mask is None)

            ap = np.vstack(apimg).T
            ap[np.logical_not(np.isfinite(ap))] = 0.
            C.AP.set('apflux_img_%s' % band, ap)
            ap = 1./(np.vstack(apimgerr).T)**2
            ap[np.logical_not(np.isfinite(ap))] = 0.
            C.AP.set('apflux_img_ivar_%s' % band, ap)
            ap = np.vstack(apmask).T
            ap[np.logical_not(np.isfinite(ap))] = 0.
            C.AP.set('apflux_masked_%s' % band, ap)
            if mods is not None:
                ap = np.vstack(apres).T
                ap[np.logical_not(np.isfinite(ap))] = 0.
                C.AP.set('apflux_resid_%s' % band, ap)
            if blobmods is not None:
                ap = np.vstack(apblobres).T
                ap[np.logical_not(np.isfinite(ap))] = 0.
                C.AP.set('apflux_blobresid_%s' % band, ap)

        t3 = Time()
        debug('coadds apphot:', t3-t2)

    return C

def _make_coadds_plots_4(allresids, mods, ps):
    import pylab as plt
    I = np.argsort([a[0] for a in allresids])
    cols = int(np.ceil(np.sqrt(len(I))))
    rows = int(np.ceil(len(I) / float(cols)))
    allresids = [allresids[i] for i in I]
    plt.clf()
    for i,(y,n,img,mod,res) in enumerate(allresids):
        plt.subplot(rows,cols,i+1)
        plt.imshow(img, interpolation='nearest', origin='lower', cmap='gray')
        plt.xticks([]); plt.yticks([])
        plt.title('%.1f: %s' % (y, n))
    plt.suptitle('Data')
    ps.savefig()
    if mods is not None:
        plt.clf()
        for i,(y,n,img,mod,res) in enumerate(allresids):
            plt.subplot(rows,cols,i+1)
            plt.imshow(mod, interpolation='nearest', origin='lower', cmap='gray')
            plt.xticks([]); plt.yticks([])
            plt.title('%.1f: %s' % (y, n))
        plt.suptitle('Model')
        ps.savefig()
        plt.clf()
        for i,(y,n,img,mod,res) in enumerate(allresids):
            plt.subplot(rows,cols,i+1)
            plt.imshow(res, interpolation='nearest', origin='lower', cmap='gray',
                       vmin=-20, vmax=20)
            plt.xticks([]); plt.yticks([])
            plt.title('%.1f: %s' % (y, n))
        plt.suptitle('Resids')
        ps.savefig()

def _make_coadds_plots_3(cowimg, cow, coimg, band, ps):
    import pylab as plt
    plt.clf()
    plt.subplot(2,2,1)
    mn,mx = cowimg.min(), cowimg.max()
    plt.imshow(cowimg, interpolation='nearest', origin='lower', cmap='gray',
               vmin=mn, vmax=mx)
    plt.xticks([]); plt.yticks([])
    plt.title('weighted img')
    plt.subplot(2,2,2)
    mycow = cow.copy()
    # mark zero as special color
    #mycow[mycow == 0] = np.nan
    plt.imshow(mycow, interpolation='nearest', origin='lower', cmap='gray',
               vmin=0)
    plt.xticks([]); plt.yticks([])
    plt.title('weights')
    plt.subplot(2,2,3)
    plt.imshow(coimg, interpolation='nearest', origin='lower', cmap='gray')
    plt.xticks([]); plt.yticks([])
    plt.title('unweighted img')
    mycowimg = cowimg.copy()
    mycowimg[cow == 0] = coimg[cow == 0]
    plt.subplot(2,2,4)
    plt.imshow(mycowimg, interpolation='nearest', origin='lower',
               cmap='gray', vmin=mn, vmax=mx)
    plt.xticks([]); plt.yticks([])
    plt.title('patched img')
    plt.suptitle('band %s' % band)
    ps.savefig()

def _make_coadds_plots_2(patch, copsf, psf_img, tim, band, ps):
    import pylab as plt
    plt.clf()
    plt.subplot(2,2,1)
    plt.imshow(patch, interpolation='nearest', origin='lower')
    plt.title('PSF')
    plt.subplot(2,2,2)
    plt.imshow(copsf, interpolation='nearest', origin='lower')
    plt.title('resampled PSF')
    plt.subplot(2,2,3)
    plt.imshow(np.atleast_2d(psf_img), interpolation='nearest', origin='lower')
    plt.title('PSF acc')
    plt.subplot(2,2,4)
    plt.imshow(psf_img + copsf/tim.sig1**2, interpolation='nearest', origin='lower')
    plt.title('PSF acc after')
    plt.suptitle('Tim %s band %s' % (tim.name, band))
    ps.savefig()

def _make_coadds_plots_1(im, band, mods, mo, iv, unweighted,
                         dq, satur_val, allresids, ps, H, W,
                         tim, Yo, Xo):
    from legacypipe.survey import get_rgb
    import pylab as plt
    # # Make one grayscale, brick-space plot per image
    # thisimg = np.zeros((H,W), np.float32)
    # thisimg[Yo,Xo] = im
    # rgb = get_rgb([thisimg], [band])
    # rgb = rgb.sum(axis=2)
    # fn = ps.getnext()
    # plt.imsave(fn, rgb, origin='lower', cmap='gray')
    #plt.clf()
    #plt.imshow(rgb, interpolation='nearest', origin='lower', cmap='gray')
    #plt.xticks([]); plt.yticks([])
    #ps.savefig()
    # Image, Model, and Resids
    plt.clf()
    plt.subplot(2,2,1)
    thisimg = np.zeros((H,W), np.float32)
    thisimg[Yo,Xo] = im
    rgb = get_rgb([thisimg], [band])
    iplane = dict(g=2, r=1, z=0)[band]
    rgbimg = rgb[:,:,iplane]
    plt.imshow(rgbimg, interpolation='nearest', origin='lower', cmap='gray')
    plt.xticks([]); plt.yticks([])
    if mods is not None:
        plt.subplot(2,2,2)
        thismod = np.zeros((H,W), np.float32)
        thismod[Yo,Xo] = mo
        rgb = get_rgb([thismod], [band])
        rgbmod = rgb[:,:,iplane]
        plt.imshow(rgbmod, interpolation='nearest', origin='lower', cmap='gray')
        plt.xticks([]); plt.yticks([])
        plt.subplot(2,2,3)
        thisres = np.zeros((H,W), np.float32)
        thisres[Yo,Xo] = (im - mo) * np.sqrt(iv)
        plt.imshow(thisres, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=-20, vmax=20)
        plt.xticks([]); plt.yticks([])
    else:
        if unweighted and (dq is not None):

            # HACK -- copy-n-pasted code from below.
            okbits = 0
            #for bitname in ['satur', 'bleed']:
            for bitname in ['satur']:
                okbits |= DQ_BITS[bitname]
            brightpix = ((dq & okbits) != 0)
            myim = im.copy()
            if satur_val is not None:
                # HACK -- force SATUR pix to be bright
                myim[brightpix] = satur_val
            #for bitname in ['interp']:
            for bitname in ['interp', 'bleed']:
                okbits |= DQ_BITS[bitname]
            goodpix = ((dq & ~okbits) == 0)
            thisgood = np.zeros((H,W), np.float32)
            thisgood[Yo,Xo] = goodpix
            plt.subplot(2,2,2)
            plt.imshow(thisgood, interpolation='nearest', origin='lower', cmap='gray', vmin=0, vmax=1)
            plt.xticks([]); plt.yticks([])
            plt.title('goodpix')

            thisim = np.zeros((H,W), np.float32)
            thisim[Yo,Xo] = goodpix * myim
            rgb = get_rgb([thisim], [band])
            iplane = dict(g=2, r=1, z=0)[band]
            rgbimg = rgb[:,:,iplane]
            plt.subplot(2,2,3)
            plt.imshow(rgbimg, interpolation='nearest', origin='lower', cmap='gray')
            plt.xticks([]); plt.yticks([])
            plt.title('goodpix rgb')


        rgbmod=None
        thisres=None

    plt.subplot(2,2,4)
    thisiv = np.zeros((H,W), np.float32)
    thisiv[Yo,Xo] = iv
    plt.imshow(thisiv, interpolation='nearest', origin='lower', cmap='gray')
    plt.xticks([]); plt.yticks([])
    plt.title('invvar')
    plt.suptitle(tim.name + ': %.2f' % (tim.time.toYear()))
    ps.savefig()
    allresids.append((tim.time.toYear(), tim.name, rgbimg,rgbmod,thisres))

def _resample_one(args):
    (itim,tim,mod,blobmod,lanczos,targetwcs,sbscale) = args
    if lanczos:
        from astrometry.util.miscutils import patch_image
        patched = tim.getImage().copy()
        assert(np.all(np.isfinite(tim.getInvError())))
        okpix = (tim.getInvError() > 0)
        patch_image(patched, okpix)
        del okpix
        imgs = [patched]
        if mod is not None:
            imgs.append(mod)
        if blobmod is not None:
            imgs.append(blobmod)
    else:
        imgs = []

    try:
        Yo,Xo,Yi,Xi,rimgs = resample_with_wcs(
            targetwcs, tim.subwcs, imgs, 3, intType=np.int16)
    except OverlapError:
        return None
    if len(Yo) == 0:
        return None
    mo = None
    bmo = None
    if lanczos:
        im = rimgs[0]
        inext = 1
        if mod is not None:
            mo = rimgs[inext]
            inext += 1
        if blobmod is not None:
            bmo = rimgs[inext]
            inext += 1
        del patched,imgs,rimgs
    else:
        im = tim.getImage ()[Yi,Xi]
        if mod is not None:
            mo = mod[Yi,Xi]
        if blobmod is not None:
            bmo = blobmod[Yi,Xi]
    iv = tim.getInvvar()[Yi,Xi]
    if sbscale:
        fscale = tim.sbscale
        debug('Applying surface-brightness scaling of %.3f to' % fscale, tim.name)
        im *=  fscale
        iv /= (fscale**2)
        if mod is not None:
            mo *= fscale
        if blobmod is not None:
            bmo *= fscale
    if tim.dq is None:
        dq = None
    else:
        dq = tim.dq[Yi,Xi]
    return itim,Yo,Xo,iv,im,mo,bmo,dq

def _apphot_one(args):
    (irad, band, rad, img, sigma, mask, isimage, apxy) = args
    import photutils
    result = [irad, band, isimage]
    aper = photutils.CircularAperture(apxy, rad)
    p = photutils.aperture_photometry(img, aper, error=sigma, mask=mask)
    result.append(p.field('aperture_sum'))
    if sigma is not None:
        result.append(p.field('aperture_sum_err'))
    else:
        result.append(None)

    # If a mask is passed, also photometer it!
    if mask is not None:
        p = photutils.aperture_photometry(mask, aper)
        maskedpix = p.field('aperture_sum')
        # normalize by number of pixels (pi * rad**2)
        maskedpix /= (np.pi * rad**2)
        result.append(maskedpix)
    else:
        result.append(None)

    return result

def write_coadd_images(band,
                       survey, brickname, version_header, tims, targetwcs,
                       co_sky,
                       cowimg=None, cow=None, cowmod=None, cochi2=None,
                       cowblobmod=None,
                       psfdetiv=None, galdetiv=None, congood=None,
                       psfsize=None, **kwargs):

    # copy version_header before modifying...
    hdr = fitsio.FITSHDR()
    for r in version_header.records():
        hdr.add_record(r)
    # Grab these keywords from all input files for this band...
    keys = ['OBSERVAT', 'TELESCOP','OBS-LAT','OBS-LONG','OBS-ELEV',
            'INSTRUME','FILTER']
    comms = ['Observatory name', 'Telescope  name', 'Latitude (deg)', 'Longitude (deg)',
             'Elevation (m)', 'Instrument name', 'Filter name']
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
            hdr.add_record(dict(name=kk, value=v[ik],comment=comms[ik]))
    hdr.add_record(dict(name='FILTERX', value=band, comment='Filter short name'))

    # DATE-OBS converted to TAI.
    mjds = [tim.time.toMjd() for tim in tims if tim.band == band]
    minmjd = min(mjds)
    maxmjd = max(mjds)
    meanmjd = np.mean(mjds)
    hdr.add_record(dict(name='MJD_MIN', value=minmjd,
                        comment='Earliest MJD in coadd (TAI)'))
    hdr.add_record(dict(name='MJD_MAX', value=maxmjd,
                        comment='Latest MJD in coadd (TAI)'))
    hdr.add_record(dict(name='MJD_MEAN', value=meanmjd,
                        comment='Mean MJD in coadd (TAI)'))
    # back to date string in UTC...
    import astropy.time
    tt = [astropy.time.Time(mjd, format='mjd', scale='tai').utc.isot
          for mjd in [minmjd, maxmjd, meanmjd]]
    hdr.add_record(dict(
        name='DATEOBS1', value=tt[0],
        comment='DATE-OBS for the first image in the stack (UTC)'))
    hdr.add_record(dict(
        name='DATEOBS2', value=tt[1],
        comment='DATE-OBS for the last  image in the stack (UTC)'))
    hdr.add_record(dict(
        name='DATEOBS', value=tt[2],
        comment='Mean DATE-OBS for the stack (UTC)'))

    # Plug the WCS header cards into these images
    targetwcs.add_to_header(hdr)
    hdr.delete('IMAGEW')
    hdr.delete('IMAGEH')
    hdr.add_record(dict(name='EQUINOX', value=2000., comment='Observation Epoch'))

    imgs = [
        ('image',  'image', cowimg),
        ('invvar', 'wtmap', cow   ),
        ]
    if congood is not None:
        imgs.append(('nexp',   'expmap',   congood))
    if psfdetiv is not None:
        imgs.append(('depth', 'psfdepth', psfdetiv))
    if galdetiv is not None:
        imgs.append(('galdepth', 'galdepth', galdetiv))
    if psfsize is not None:
        imgs.append(('psfsize', 'psfsize', psfsize))
    if cowmod is not None:
        imgs.extend([
                ('model', 'model', cowmod),
                ('chi2',  'chi2',  cochi2),
                ])
    if cowblobmod is not None:
        imgs.append(('blobmodel', 'blobmodel', cowblobmod))
    for name,prodtype,img in imgs:
        from legacypipe.survey import MyFITSHDR
        if img is None:
            debug('Image type', prodtype, 'is None -- skipping')
            continue
        # Make a copy, because each image has different values for
        # these headers...
        hdr2 = MyFITSHDR()
        for r in hdr.records():
            hdr2.add_record(r)
        hdr2.add_record(dict(name='IMTYPE', value=name,
                             comment='LegacySurveys image type'))
        hdr2.add_record(dict(name='PRODTYPE', value=prodtype,
                             comment='NOAO image type'))
        if name in ['image', 'model', 'blobmodel']:
            hdr2.add_record(dict(name='MAGZERO', value=22.5,
                                 comment='Magnitude zeropoint'))
            hdr2.add_record(dict(name='BUNIT', value='nanomaggy',
                                 comment='AB mag = 22.5 - 2.5*log10(nanomaggy)'))
        if name == 'image' and co_sky is not None:
            hdr2.add_record(dict(name='COSKY_%s' % band.upper(), value=co_sky.get(band, 'None'),
                                 comment='Sky level estimated (+subtracted) from coadd'))
        if name in ['invvar', 'depth', 'galdepth']:
            hdr2.add_record(dict(name='BUNIT', value='1/nanomaggy^2',
                                 comment='Ivar of ABmag=22.5-2.5*log10(nmgy)'))
        if name in ['psfsize']:
            hdr2.add_record(dict(name='BUNIT', value='arcsec',
                                 comment='Effective PSF size'))
        with survey.write_output(name, brick=brickname, band=band,
                                 shape=img.shape) as out:
            out.fits.write(img, header=hdr2)

# Pretty much only used for plots; the real deal is make_coadds()
def quick_coadds(tims, bands, targetwcs, images=None,
                 get_cow=False, get_n2=False, fill_holes=True, get_max=False):

    W = int(targetwcs.get_width())
    H = int(targetwcs.get_height())

    coimgs = []
    cons = []
    if get_n2:
        cons2 = []
    if get_cow:
        # moo
        cowimgs = []
        wimgs = []
    if get_max:
        maximgs = []

    for band in bands:
        coimg  = np.zeros((H,W), np.float32)
        coimg2 = np.zeros((H,W), np.float32)
        con    = np.zeros((H,W), np.int16)
        con2   = np.zeros((H,W), np.int16)
        if get_cow:
            cowimg = np.zeros((H,W), np.float32)
            wimg   = np.zeros((H,W), np.float32)
        if get_max:
            maximg = np.zeros((H,W), np.float32)

        for itim,tim in enumerate(tims):
            if tim.band != band:
                continue
            R = tim_get_resamp(tim, targetwcs)
            if R is None:
                continue
            (Yo,Xo,Yi,Xi) = R
            nn = (tim.getInvError()[Yi,Xi] > 0)
            if images is None:
                coimg [Yo,Xo] += tim.getImage()[Yi,Xi] * nn
                coimg2[Yo,Xo] += tim.getImage()[Yi,Xi]
                if get_max:
                    maximg[Yo,Xo] = np.maximum(maximg[Yo,Xo], tim.getImage()[Yi,Xi] * nn)
            else:
                coimg [Yo,Xo] += images[itim][Yi,Xi] * nn
                coimg2[Yo,Xo] += images[itim][Yi,Xi]
                if get_max:
                    maximg[Yo,Xo] = np.maximum(maximg[Yo,Xo], images[itim][Yi,Xi] * nn)
            con   [Yo,Xo] += nn
            if get_cow:
                cowimg[Yo,Xo] += tim.getInvvar()[Yi,Xi] * tim.getImage()[Yi,Xi]
                wimg  [Yo,Xo] += tim.getInvvar()[Yi,Xi]
            con2  [Yo,Xo] += 1
        coimg /= np.maximum(con,1)
        if fill_holes:
            coimg[con == 0] = coimg2[con == 0] / np.maximum(1, con2[con == 0])
        if get_cow:
            cowimg /= np.maximum(wimg, 1e-16)
            cowimg[wimg == 0] = coimg[wimg == 0]
            cowimgs.append(cowimg)
            wimgs.append(wimg)
        if get_max:
            maximgs.append(maximg)
        coimgs.append(coimg)
        cons.append(con)
        if get_n2:
            cons2.append(con2)

    rtn = [coimgs,cons]
    if get_cow:
        rtn.extend([cowimgs, wimgs])
    if get_n2:
        rtn.append(cons2)
    if get_max:
        rtn.append(maximgs)
    return rtn
