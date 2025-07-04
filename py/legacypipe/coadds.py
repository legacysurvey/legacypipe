import numpy as np
import fitsio
from astrometry.util.fits import fits_table
from astrometry.util.resample import resample_with_wcs, OverlapError
from legacypipe.bits import DQ_BITS
from legacypipe.survey import tim_get_resamp
from legacypipe.utils import copy_header_with_wcs

import logging
logger = logging.getLogger('legacypipe.coadds')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

class SimpleCoadd(object):
    '''A class for handling coadds of unWISE (and GALEX) images.
    '''
    def __init__(self, ra, dec, W, H, pixscale, bands):
        from legacypipe.survey import wcs_for_brick, BrickDuck
        self.W = W
        self.H = H
        self.bands = bands
        brick = BrickDuck(ra, dec, 'quack')
        self.wcs = wcs_for_brick(brick, W=self.W, H=self.H,
                                        pixscale=pixscale)
        # images
        self.co_images = dict([(band, np.zeros((self.H,self.W), np.float32))
                               for band in bands])
        self.co_nobs = dict([(band, np.zeros((self.H,self.W), np.uint16))
                             for band in bands])
        # models
        self.co_models = dict([(band, np.zeros((self.H,self.W), np.float32))
                               for band in bands])
        # invvars
        self.co_invvars = dict([(band, np.zeros((self.H,self.W), np.float32))
                                for band in bands])

    def add(self, models, unique=False):
        for name, band, wcs, img, mod, ie in models:
            debug('Accumulating tile', name, 'band', band)
            try:
                Yo,Xo,Yi,Xi,resam = resample_with_wcs(self.wcs, wcs,
                                                      [img, mod], intType=np.int16)
            except OverlapError:
                debug('No overlap between tile', name, 'and coadd')
                continue
            rimg,rmod = resam
            debug('Adding', len(Yo), 'pixels from tile', name, 'to coadd')
            iv = ie[Yi,Xi]**2
            if unique:
                K = np.flatnonzero((self.co_nobs[band][Yo,Xo] == 0) * (iv>0))
                iv = iv[K]
                rimg = rimg[K]
                rmod = rmod[K]
                Yo = Yo[K]
                Xo = Xo[K]
                debug('Cut to', len(Yo), 'unique pixels w/ iv>0')

            debug('Tile:', np.sum(iv>0), 'of', len(iv), 'pixels have IV')
            self.co_images [band][Yo,Xo] += rimg * iv
            self.co_models [band][Yo,Xo] += rmod * iv
            self.co_nobs   [band][Yo,Xo] += 1
            self.co_invvars[band][Yo,Xo] += iv
            debug('Band', band, ': now', np.sum(self.co_nobs[band]>0), 'pixels are set in image coadd')

    def finish(self, survey, brickname, version_header,
               apradec=None, apertures=None):
        # apradec = (ra,dec): aperture photometry locations
        # apertures: RADII in PIXELS
        if apradec is not None:
            assert(apertures is not None)
            (ra,dec) = apradec
            ok,xx,yy = self.wcs.radec2pixelxy(ra, dec)
            assert(np.all(ok))
            del ok
            apxy = np.vstack((xx - 1., yy - 1.)).T
            ap_iphots = [np.zeros((len(ra), len(apertures)), np.float32)
                         for band in self.bands]
            ap_dphots = [np.zeros((len(ra), len(apertures)), np.float32)
                         for band in self.bands]
            ap_rphots = [np.zeros((len(ra), len(apertures)), np.float32)
                         for band in self.bands]

        coimgs = []
        comods = []
        for iband,band in enumerate(self.bands):
            coimg = self.co_images[band]
            comod = self.co_models[band]
            coiv  = self.co_invvars[band]
            con   = self.co_nobs[band]
            with np.errstate(divide='ignore', invalid='ignore'):
                coimg /= coiv
                comod /= coiv
            coimg[coiv == 0] = 0.
            comod[coiv == 0] = 0.
            coimgs.append(coimg)
            comods.append(comod)

            hdr = copy_header_with_wcs(version_header, self.wcs)
            self.add_to_header(hdr, band)
            self.write_coadds(survey, brickname, hdr, band, coimg, comod, coiv, con)

            if apradec is not None:
                from photutils.aperture import CircularAperture, aperture_photometry
                mask = (coiv == 0)
                with np.errstate(divide='ignore'):
                    imsigma = 1.0/np.sqrt(coiv)
                imsigma[mask] = 0.
                for irad,rad in enumerate(apertures):
                    aper = CircularAperture(apxy, rad)
                    p = aperture_photometry(coimg, aper, error=imsigma, mask=mask)
                    ap_iphots[iband][:,irad] = p.field('aperture_sum')
                    ap_dphots[iband][:,irad] = p.field('aperture_sum_err')
                    p = aperture_photometry(coimg - comod, aper, mask=mask)
                    ap_rphots[iband][:,irad] = p.field('aperture_sum')

        self.write_color_image(survey, brickname, coimgs, comods)

        if apradec is not None:
            return ap_iphots, ap_dphots, ap_rphots

    def add_to_header(self, hdr, band):
        pass

    def write_coadds(self, survey, brickname, hdr, band, coimg, comod, coiv, con):
        pass

    def write_color_image(self, survey, brickname, coimgs, comods):
        pass

class UnwiseCoadd(SimpleCoadd):
    def __init__(self, ra, dec, W, H, pixscale):
        super().__init__(ra, dec, W, H, pixscale, [1,2,3,4])

    def add_to_header(self, hdr, band):
        hdr.add_record(dict(name='TELESCOP', value='WISE'))
        hdr.add_record(dict(name='FILTER', value='W%i' % band, comment='WISE band'))
        hdr.add_record(dict(name='MAGZERO', value=22.5,
                            comment='Magnitude zeropoint'))
        hdr.add_record(dict(name='MAGSYS', value='Vega',
                            comment='This WISE image is in Vega fluxes'))

    def write_coadds(self, survey, brickname, hdr, band, coimg, comod, coiv, con):
        with survey.write_output('image', brick=brickname, band='W%i'%band,
                                 shape=coimg.shape) as out:
            out.fits.write(coimg, header=hdr, extname='IMAGE_W%s' % band)
        with survey.write_output('model', brick=brickname, band='W%i'%band,
                                 shape=comod.shape) as out:
            out.fits.write(comod, header=hdr, extname='MODEL_W%s' % band)
        with survey.write_output('invvar', brick=brickname, band='W%i'%band,
                                 shape=coiv.shape) as out:
            out.fits.write(coiv, header=hdr, extname='INVVAR_W%s' % band)

    def write_color_image(self, survey, brickname, coimgs, comods):
        from legacypipe.survey import imsave_jpeg
        # W1/W2 color jpeg
        rgb = _unwise_to_rgb(coimgs[:2])
        with survey.write_output('wise-jpeg', brick=brickname) as out:
            imsave_jpeg(out.fn, rgb, origin='lower')
            info('Wrote', out.fn)
        rgb = _unwise_to_rgb(comods[:2])
        with survey.write_output('wisemodel-jpeg', brick=brickname) as out:
            imsave_jpeg(out.fn, rgb, origin='lower')
            info('Wrote', out.fn)
        coresids = [coimg - comod for coimg, comod in zip(coimgs[:2], comods[:2])]
        rgb = _unwise_to_rgb(coresids)
        with survey.write_output('wiseresid-jpeg', brick=brickname) as out:
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
        with np.errstate(divide='ignore'):
            img1 = I * img1 / mean
            img2 = I * img2 / mean
        img1[mean == 0] = 0.
        img2[mean == 0] = 0.
        mn = nlmap(mn)
        mx = nlmap(mx)
    img1 = (img1 - mn) / (mx - mn)
    img2 = (img2 - mn) / (mx - mn)
    rgb[:,:,2] = (np.clip(img1, 0., 1.) * 255).astype(np.uint8)
    rgb[:,:,0] = (np.clip(img2, 0., 1.) * 255).astype(np.uint8)
    rgb[:,:,1] = rgb[:,:,0]/2 + rgb[:,:,2]/2
    return rgb

class Coadd(object):
    def __init__(self, band, H, W, detmaps, mods, blobmods, unweighted, ngood,
                 xy, allmasks, anymasks, nsatur, psfsize, do_max, psf_images,
                 satur_val, targetwcs):
        # coadded weight map (moo)
        self.cow    = np.zeros((H,W), np.float32)
        # coadded weighted image map
        self.cowimg = np.zeros((H,W), np.float32)
        self.kwargs = dict(cowimg=self.cowimg, cow=self.cow)

        # We have *three* counters for the number of pixels
        # overlapping each coadd brick pixel:
        #
        # - "con" counts the pixels included in the unweighted coadds.
        #   This map is not passed outside this function or used
        #   anywhere else.
        #
        # - "congood" counts pixels with (iv > 0).  This gets passed
        #   to the *write_coadd_images* function, where it gets
        #   written to the *nexp* maps.
        #
        # - "nobs" counts all pixels, regardless of masks.  This gets
        #   sampled at *xy* positions, and ends up in the tractor
        #   catalog "nobs" column.
        #
        # (you want to know the number of observations within the
        # source footprint, not just the peak pixel which may be
        # saturated, etc.)

        if detmaps:
            # detection map inverse-variance (depth map)
            self.psfdetiv = np.zeros((H,W), np.float32)
            # galaxy detection map inverse-variance (galdepth map)
            self.galdetiv = np.zeros((H,W), np.float32)
            self.kwargs.update(psfdetiv=self.psfdetiv,
                               galdetiv=self.galdetiv)

        if mods:
            # model image
            self.cowmod = np.zeros((H,W), np.float32)
            # chi-squared image
            self.cochi2 = np.zeros((H,W), np.float32)
            self.kwargs.update(cowmod=self.cowmod, cochi2=self.cochi2)

        if blobmods:
            # model image
            self.cowblobmod = np.zeros((H,W), np.float32)
            self.kwargs.update(cowblobmod=self.cowblobmod)

        if unweighted:
            # unweighted image
            self.coimg  = np.zeros((H,W), np.float32)
            if mods:
                # unweighted model
                self.comod  = np.zeros((H,W), np.float32)
            if blobmods:
                self.coblobmod  = np.zeros((H,W), np.float32)
            # number of exposures
            self.con = np.zeros((H,W), np.int16)
            self.kwargs.update(coimg=self.coimg)

        if ngood:
            self.congood = np.zeros((H,W), np.int16)
            self.kwargs.update(congood=self.congood)

        # FIXME - DQ datatype!!

        if xy or allmasks or anymasks:
            # These match the type of the "DQ" images.
            # "any" mask
            self.ormask  = np.zeros((H,W), np.int16)
            # "all" mask
            self.andmask = np.empty((H,W), np.int16)
            from functools import reduce
            allbits = reduce(np.bitwise_or, DQ_BITS.values())
            self.andmask[:,:] = allbits
            self.kwargs.update(ormask=self.ormask, andmask=self.andmask)

        if xy or allmasks:
            # number of observations
            self.nobs = np.zeros((H,W), np.int16)

        if nsatur:
            self.satmap = np.zeros((H,W), np.int16)

        if psfsize:
            self.psfsizemap = np.zeros((H,W), np.float32)
            # like "cow", but constant invvar per-CCD;
            # only required for psfsizemap
            self.flatcow = np.zeros((H,W), np.float32)
            self.kwargs.update(psfsize=self.psfsizemap)

        if do_max:
            self.maximg = np.zeros((H,W), np.float32)

        if psf_images:
            self.psf_img = 0.

        self.unweighted = unweighted
        self.satur_val = satur_val
        self.xy = xy
        self.allmasks = allmasks
        self.anymasks = anymasks
        self.nsatur = nsatur
        self.psfsize = psfsize
        self.psf_images = psf_images
        self.detmaps = detmaps
        self.ngood = ngood
        self.mods = mods
        self.blobmods = blobmods
        self.do_max = do_max
        self.targetwcs = targetwcs

    def accumulate(self, tim, itim, Yo,Xo,iv,im,mo,bmo,dq, mjd_args):
        # invvar-weighted image
        self.cowimg[Yo,Xo] += iv * im
        self.cow   [Yo,Xo] += iv

        goodpix = None
        if self.unweighted:
            if dq is None:
                goodpix = 1
            else:
                # FIXME -- DQ datatype?
                # include SATUR pixels if no other pixels exists
                okbits = np.uint16(0)
                for bitname in ['satur']:
                    okbits |= DQ_BITS[bitname]
                brightpix = ((dq & okbits) != 0)
                if self.satur_val is not None:
                    # HACK -- force SATUR pix to be bright
                    im[brightpix] = self.satur_val
                # Include these pixels if none other exist??
                for bitname in ['interp']:
                    okbits |= DQ_BITS[bitname]
                goodpix = ((dq & ~okbits) == 0)

            self.coimg[Yo,Xo] += goodpix * im
            self.con  [Yo,Xo] += goodpix

        if self.xy or self.allmasks or self.anymasks:
            if dq is not None:
                self.ormask [Yo,Xo] |= dq
                self.andmask[Yo,Xo] &= dq
        if self.xy or self.allmasks:
            # raw exposure count
            self.nobs[Yo,Xo] += 1
        if self.xy and (mjd_args is not None):
            mjds, mjd_argmins, mjd_argmaxs = mjd_args
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

        # FIXME  - satur_bits ???

        if self.nsatur and dq is not None:
            self.satmap[Yo,Xo] += (1*((dq & DQ_BITS['satur'])>0))

        if self.psfsize:
            # psfnorm is in units of 1/pixels.
            # (eg, psfnorm for a gaussian is 1./(2.*sqrt(pi) * psf_sigma) )
            # Neff is in pixels**2
            neff = 1./tim.psfnorm**2
            # Narcsec is in arcsec**2
            narcsec = neff * tim.wcs.pixel_scale()**2
            # Make smooth maps -- don't ignore CRs, saturated pix, etc
            iv1 = 1./tim.sig1**2
            self.psfsizemap[Yo,Xo] += iv1 * (1. / narcsec)
            self.flatcow   [Yo,Xo] += iv1

        if self.psf_images:
            from astrometry.util.util import lanczos3_interpolate
            h,w = tim.shape
            patch = tim.psf.getPointSourcePatch(w//2, h//2).patch
            patch /= np.sum(patch)
            # In case the tim and coadd have different pixel scales,
            # resample the PSF stamp.
            ph,pw = patch.shape
            pscale = tim.imobj.pixscale / self.targetwcs.pixel_scale()
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
            lanczos3_interpolate(ix, iy, dx, dy, [copsf], [patch])
            copsf = copsf.reshape((coph,copw))
            copsf /= copsf.sum()
            #if plots:
            #    _make_coadds_plots_2(patch, copsf, psf_img, tim, band, ps)
            self.psf_img = self.psf_img + copsf / tim.sig1**2

        if self.detmaps:
            # point-source depth
            detsig1 = tim.sig1 / tim.psfnorm
            self.psfdetiv[Yo,Xo] += (iv > 0) * (1. / detsig1**2)
            # Galaxy detection map
            gdetsig1 = tim.sig1 / tim.galnorm
            self.galdetiv[Yo,Xo] += (iv > 0) * (1. / gdetsig1**2)

        if self.ngood:
            self.congood[Yo,Xo] += (iv > 0)

        if self.mods:
            # I think we always require unweighted = True
            assert(goodpix is not None)
            if mo is not None:
                # straight-up
                self.comod[Yo,Xo] += goodpix * mo
                # invvar-weighted
                self.cowmod[Yo,Xo] += iv * mo
                # chi-squared
                self.cochi2[Yo,Xo] += iv * (im - mo)**2
            del mo

        if self.blobmods:
            # straight-up
            self.coblobmod[Yo,Xo] += goodpix * bmo
            # invvar-weighted
            self.cowblobmod[Yo,Xo] += iv * bmo
            del bmo
        del goodpix

        if self.do_max:
            self.maximg[Yo,Xo] = np.maximum(self.maximg[Yo,Xo], im * (iv>0))

    def finish(self):
        tinyw = 1e-30
        self.cowimg /= np.maximum(self.cow, tinyw)
        if self.mods:
            self.cowmod /= np.maximum(self.cow, tinyw)
            self.coresid = self.cowimg - self.cowmod
            self.coresid[self.cow == 0] = 0.

        if self.blobmods:
            self.cowblobmod  /= np.maximum(self.cow, tinyw)
            self.coblobresid = self.cowimg - self.cowblobmod
            self.coblobresid[self.cow == 0] = 0.

        if self.xy or self.allmasks:
            # If there was no coverage, don't set ALLMASK
            self.andmask[self.nobs == 0] = 0

        if self.nsatur:
            self.satmap = (self.satmap >= nsatur)

        if self.psf_images:
            self.psf_img /= np.sum(self.psf_img)

        if self.unweighted:
            self.coimg  /= np.maximum(self.con, 1)
            del self.con

            #if plots:
            #    _make_coadds_plots_3(cowimg, cow, coimg, band, ps)

            # Patch pixels with no data in the weighted coadd.
            self.cowimg[self.cow == 0] = self.coimg[self.cow == 0]
            del self.coimg
            if self.mods:
                self.cowmod[self.cow == 0] = self.comod[self.cow == 0]
                del self.comod
            if self.blobmods:
                self.cowblobmod[self.cow == 0] = self.coblobmod[self.cow == 0]
                del self.coblobmod

        if self.psfsize:
            # psfsizemap is accumulated in units of iv * (1 / arcsec**2)
            # take out the weighting
            self.psfsizemap /= np.maximum(self.flatcow, tinyw)
            # Correction factor to get back to equivalent of Gaussian sigma
            tosigma = 1./(2. * np.sqrt(np.pi))
            # Conversion factor to FWHM (2.35)
            tofwhm = 2. * np.sqrt(2. * np.log(2.))
            # Scale back to units of linear arcsec.
            with np.errstate(divide='ignore'):
                self.psfsizemap[:,:] = (1. / np.sqrt(self.psfsizemap)) * tosigma * tofwhm
            self.psfsizemap[self.flatcow == 0] = 0.

def make_coadds(tims, bands, targetwcs,
                coweights=True,
                mods=None, blobmods=None,
                xy=None, apertures=None, apxy=None,
                ngood=False, detmaps=False, psfsize=False,
                allmasks=True, anymasks=False,
                mjdminmax=True,
                get_max=False, sbscale=True,
                psf_images=False, nsatur=None,
                callback=None, callback_args=None,
                plots=False, ps=None,
                lanczos=True, mp=None,
                satur_val=10.):
    from astrometry.util.ttime import Time
    t0 = Time()

    if callback_args is None:
        callback_args = []

    # This is the object that will be returned by this method
    class Duck(object):
        pass
    C = Duck()

    W = int(targetwcs.get_width())
    H = int(targetwcs.get_height())

    # always, for patching SATUR, etc pixels?
    unweighted=True

    C.coimgs = []
    if coweights:
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
    if anymasks:
        C.anymasks = []
    if max:
        C.maximgs = []
    if psf_images:
        C.psf_imgs = []
    if nsatur:
        C.satmaps = []

    if xy:
        ix,iy = xy
        C.T = fits_table()
        if ngood:
            C.T.ngood   = np.zeros((len(ix), len(bands)), np.int16)
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

    mjd_args = None
    if xy and mjdminmax:
        # To save the memory of 2 x float64 maps, we instead do arg min/max maps

        # append a 0 to the list of mjds so that mjds[-1] gives 0.
        mjds = np.array([tim.time.toMjd() for tim in tims] + [0])
        mjd_argmins = np.empty((H,W), np.int16)
        mjd_argmaxs = np.empty((H,W), np.int16)
        mjd_argmins[:,:] = -1
        mjd_argmaxs[:,:] = -1
        mjd_args = (mjds, mjd_argmins, mjd_argmaxs)

    if plots:
        allresids = []

    for iband,(band,timiter) in enumerate(zip(bands, imaps)):
        debug('Computing coadd for band', band)

        coadd = Coadd(band, H, W, detmaps,
                      (mods is not None), (blobmods is not None), unweighted, ngood,
                      xy, allmasks, anymasks, nsatur, psfsize, max, psf_images,
                      satur_val, targetwcs)

        for R in timiter:
            if R is None:
                continue
            itim,Yo,Xo,iv,im,mo,bmo,dq = R
            tim = tims[itim]

            if plots:
                _make_coadds_plots_1(im, band, mods, mo, iv, unweighted,
                                     dq, satur_val, allresids, ps, H, W,
                                     tim, Yo, Xo)

            coadd.accumulate(tim, itim,Yo,Xo,iv,im,mo,bmo,dq,
                             mjd_args)

            del Yo,Xo,iv,im,mo,bmo,dq,R

        coadd.finish()

        C.coimgs.append(coadd.cowimg)
        if coweights:
            C.cowimgs.append(coadd.cow)
        if mods is not None:
            C.comods.append(coadd.cowmod)
            C.coresids.append(coadd.coresid)
        if blobmods is not None:
            C.coblobmods.append(coadd.cowblobmod)
            C.coblobresids.append(coadd.coblobresid)
        if detmaps:
            C.psfdetivs.append(coadd.psfdetiv)
            C.galdetivs.append(coadd.galdetiv)
        if max:
            C.maximgs.append(coadd.maximg)
        if allmasks:
            C.allmasks.append(coadd.andmask)
        if anymasks:
            C.anymasks.append(coadd.ormask)
        if nsatur:
            C.satmaps.append(coadd.satmap)
        if psf_images:
            C.psf_imgs.append(coadd.psf_img)
        if xy:
            C.T.nobs   [:,iband] = coadd.nobs   [iy,ix]
            C.T.anymask[:,iband] = coadd.ormask [iy,ix]
            C.T.allmask[:,iband] = coadd.andmask[iy,ix]
            if ngood:
                C.T.ngood[:,iband] = coadd.congood[iy,ix]
            if detmaps:
                C.T.psfdepth[:,iband] = coadd.psfdetiv[iy, ix]
                C.T.galdepth[:,iband] = coadd.galdetiv[iy, ix]
            if psfsize:
                C.T.psfsize[:,iband] = coadd.psfsizemap[iy,ix]

        if apertures is not None:
            # Aperture photometry
            # photutils.aperture_photometry: mask=True means IGNORE
            mask = (coadd.cow == 0)
            with np.errstate(divide='ignore'):
                imsigma = 1.0/np.sqrt(coadd.cow)
            imsigma[mask] = 0.

            for irad,rad in enumerate(apertures):
                apargs.append((irad, band, rad, coadd.cowimg, imsigma, mask,
                               True, apxy))
                if mods is not None:
                    apargs.append((irad, band, rad, coadd.coresid, None, None,
                                   False, apxy))
                if blobmods is not None:
                    apargs.append((irad, band, rad, coadd.coblobresid, None, None,
                                   False, apxy))

        if callback is not None:
            callback(band, *callback_args, **coadd.kwargs)

        del coadd
        # END of loop over bands

    t2 = Time()
    debug('coadds: images:', t2-t0)

    if plots:
        _make_coadds_plots_4(allresids, mods, ps)

    if xy and mjdminmax:
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
            with np.errstate(divide='ignore'):
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
    iplane = dict(g=2, r=1, i=0, z=0).get(band, 1)
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
            okbits = np.uint16(0)
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
            iplane = dict(g=2, r=1, i=1, z=0).get(band,1)
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
    from photutils.aperture import CircularAperture, aperture_photometry
    result = [irad, band, isimage]
    aper = CircularAperture(apxy, rad)
    p = aperture_photometry(img, aper, error=sigma, mask=mask)
    result.append(p.field('aperture_sum'))
    if sigma is not None:
        result.append(p.field('aperture_sum_err'))
    else:
        result.append(None)

    # If a mask is passed, also photometer it!
    if mask is not None:
        p = aperture_photometry(mask, aper)
        maskedpix = p.field('aperture_sum')
        # normalize by number of pixels (pi * rad**2)
        maskedpix /= (np.pi * rad**2)
        result.append(maskedpix)
    else:
        result.append(None)

    return result

def get_coadd_headers(hdr, tims, band, coadd_headers=None):
    if coadd_headers is None:
        coadd_headers = {}
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

    # add more info from fit_on_coadds
    if bool(coadd_headers):
        for key in sorted(coadd_headers.keys()):
            hdr.add_record(dict(name=key, value=coadd_headers[key][0], comment=coadd_headers[key][1]))

def write_coadd_images(band,
                       survey, brickname, version_header, tims, targetwcs,
                       co_sky,
                       coadd_headers=None,
                       cowimg=None, cow=None, cowmod=None, cochi2=None,
                       cowblobmod=None,
                       psfdetiv=None, galdetiv=None, congood=None,
                       psfsize=None, **kwargs):

    hdr = copy_header_with_wcs(version_header, targetwcs)
    # Grab headers from input images...
    get_coadd_headers(hdr, tims, band, coadd_headers)
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
        if img is None:
            debug('Image type', prodtype, 'is None -- skipping')
            continue
        # Make a copy, because each image has different values for
        # these headers...
        hdr2 = fitsio.FITSHDR()
        for r in hdr.records():
            hdr2.add_record(r)
        hdr2.add_record(dict(name='IMTYPE', value=name,
                             comment='LegacySurveys image type'))
        hdr2.add_record(dict(name='PRODTYPE', value=prodtype,
                             comment='NOIRLab image type'))
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
                                 comment='Inverse-variance maps; ABmag=22.5-2.5*log10(nmgy)'))
        if name in ['psfsize']:
            hdr2.add_record(dict(name='BUNIT', value='arcsec',
                                 comment='Effective PSF size'))
        extname = '%s_%s' % (name.upper(), band)
        with survey.write_output(name, brick=brickname, band=band,
                                 shape=img.shape) as out:
            out.fits.write(img, header=hdr2, extname=extname)

# Pretty much only used for plots; the real deal is make_coadds()
def quick_coadds(tims, bands, targetwcs, images=None,
                 get_cow=False, get_n2=False, fill_holes=True, get_max=False,
                 get_saturated=False,
                 addnoise=False):
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
    if get_saturated:
        satur = np.zeros((H,W), bool)
    if addnoise:
        noise = np.zeros((H,W), np.float32)

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
                if addnoise:
                    noise[:,:] = 0.
                    noise[Yo[nn],Xo[nn]] = 1./(tim.getInvError()[Yi,Xi][nn])
                    coimg += noise * np.random.normal(size=noise.shape)
            con   [Yo,Xo] += nn
            if get_cow:
                cowimg[Yo,Xo] += tim.getInvvar()[Yi,Xi] * tim.getImage()[Yi,Xi]
                wimg  [Yo,Xo] += tim.getInvvar()[Yi,Xi]
            if get_saturated and tim.dq is not None:
                satur[Yo,Xo] |= ((tim.dq[Yi,Xi] & tim.dq_saturation_bits) > 0)
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
    if get_saturated:
        rtn.append(satur)
    return rtn
