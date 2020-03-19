import numpy as np

def subtract_halos(tims, refs, bands, mp, plots, ps, moffat=True):
    args = [(tim, refs, moffat) for tim in tims]
    haloimgs = mp.map(subtract_one, args)
    for tim,h in zip(tims, haloimgs):
        tim.data -= h

def subtract_one(X):
    try:
        return subtract_one_real(X)
    except:
        import traceback
        traceback.print_exc()
        raise

def subtract_one_real(X):
    tim, refs, moffat = X
    if tim.imobj.camera != 'decam':
        print('Warning: Stellar halo subtraction is only implemented for DECam')
        return 0.
    return decam_halo_model(refs, tim.time.toMjd(), tim.subwcs,
                            tim.imobj.pixscale, tim.band, tim.imobj, moffat,
                            image=tim.getImage())

def moffat(rr, alpha, beta):
    return (beta-1.)/(np.pi * alpha**2)*(1. + (rr/alpha)**2)**(-beta)

def decam_halo_model(refs, mjd, wcs, pixscale, band, imobj, include_moffat,
                     image=None):

    debug = True
    
    from legacypipe.survey import radec_at_mjd
    assert(np.all(refs.ref_epoch > 0))
    rr,dd = radec_at_mjd(refs.ra, refs.dec, refs.ref_epoch.astype(float),
                         refs.pmra, refs.pmdec, refs.parallax, mjd)
    mag = refs.get('decam_mag_%s' % band)
    fluxes = 10.**((mag - 22.5) / -2.5)

    have_inner_moffat = False
    if include_moffat:
        psf = imobj.read_psf_model(0,0, pixPsf=True)
        if hasattr(psf, 'moffat'):
            have_inner_moffat = True
            inner_alpha, inner_beta = psf.moffat
            print('Read inner Moffat parameters', (inner_alpha, inner_beta),
                  'from PsfEx file')

    if debug:
        from astrometry.util.fits import fits_table
        D = refs.copy()
        D.mag = mag.copy()
        D.flux = fluxes.copy()
        D.halo_radius_pix = np.zeros(len(D), np.float32)
        D.outer_moffat_a = np.zeros(len(D), np.float32)
        D.outer_moffat_b = np.zeros(len(D), np.float32)
        D.outer_moffat_w = np.zeros(len(D), np.float32)
        D.outer_powerlaw_w = np.zeros(len(D), np.float32)
        D.inner_moffat_a = np.zeros(len(D), np.float32)
        D.inner_moffat_b = np.zeros(len(D), np.float32)
        D.x_in_sub_ccd = np.zeros(len(D), np.int16)
        D.y_in_sub_ccd = np.zeros(len(D), np.int16)

    H,W = wcs.shape
    H = int(H)
    W = int(W)
    halo = np.zeros((H,W), np.float32)
    for iref,(ref,flux,ra,dec) in enumerate(zip(refs, fluxes, rr, dd)):
        _,x,y = wcs.radec2pixelxy(ra, dec)
        x -= 1.
        y -= 1.

        if (band == 'z') and (x < 0 or y < 0 or x > W-1 or y > H-1):
            # Do not subtract z-band halos that are off the chip.
            continue

        rad_arcsec = ref.radius * 3600.
        ### FIXME -- we're going to try subtracting the halo out to
        ### TWICE our masking radius.
        rad_arcsec *= 2.0
        # Rongpu says only apply within:
        rad_arcsec = np.minimum(rad_arcsec, 400.)
        pixrad = int(np.ceil(rad_arcsec / pixscale))

        if debug:
            D.x_in_sub_ccd[iref] = int(np.round(x))
            D.y_in_sub_ccd[iref] = int(np.round(y))
            D.halo_radius_pix[iref] = pixrad

        xlo = int(np.clip(np.floor(x - pixrad), 0, W-1))
        xhi = int(np.clip(np.ceil (x + pixrad), 0, W-1))
        ylo = int(np.clip(np.floor(y - pixrad), 0, H-1))
        yhi = int(np.clip(np.ceil (y + pixrad), 0, H-1))
        if xlo == xhi or ylo == yhi:
            continue

        rads = np.hypot(np.arange(ylo, yhi+1)[:,np.newaxis] - y,
                        np.arange(xlo, xhi+1)[np.newaxis,:] - x)
        maxr = pixrad
        # Outer apodization
        apr = maxr*0.9
        apodize = np.clip((rads - maxr) / (apr - maxr), 0., 1.)

        # Inner apodization: ramp from 0 up to 1 between Rongpu's "R3"
        # and "R4" radii
        apr_i0 = 7. / pixscale
        apr_i1 = 8. / pixscale
        apodize *= np.clip((rads - apr_i0) / (apr_i1 - apr_i0), 0., 1.)

        if band == 'z':
            '''
            For z band, the outer PSF is a weighted Moffat profile. For most
            CCDs, the Moffat parameters (with radius in arcsec and SB in nmgy per
            sq arcsec) and the weight are (for a 22.5 magnitude star):
                alpha, beta, weight = 17.650, 1.7, 0.0145

            However, a small subset of DECam CCDs (which are N20, S8,
            S10, S18, S21 and S27) have a more compact outer PSF in z
            band, which can still be characterized by a weigthed
            Moffat with the following parameters:
                alpha, beta, weight = 16, 2.3, 0.0095
            '''
            if imobj.ccdname.strip() in ['N20', 'S8', 'S10', 'S18', 'S21', 'S27']:
                alpha, beta, weight = 16, 2.3, 0.0095
            else:
                alpha, beta, weight = 17.650, 1.7, 0.0145

            # The 'pixscale**2' is because Rongpu's formula is in nanomaggies/arcsec^2
            halo[ylo:yhi+1, xlo:xhi+1] += (flux * apodize * weight *
                                           moffat(rads*pixscale, alpha, beta) * pixscale**2)

            if debug:
                D.outer_moffat_a[iref] = alpha
                D.outer_moffat_b[iref] = beta
                D.outer_moffat_w[iref] = weight

        else:
             fd = dict(g=0.00045,
                       r=0.00033)
             f = fd[band]

             halo[ylo:yhi+1, xlo:xhi+1] += (flux * apodize * f * (rads*pixscale)**-2
                                            * pixscale**2)

             if debug:
                 D.outer_powerlaw_w[iref] = f
             
        if have_inner_moffat:
            weight = 1.
            halo[ylo:yhi+1, xlo:xhi+1] += (flux * apodize * weight *
                                           moffat(rads*pixscale, inner_alpha, inner_beta) * pixscale**2)
            if debug:
                D.inner_moffat_a[iref] = inner_alpha
                D.inner_moffat_b[iref] = inner_beta

    if debug:
        fn = 'halo-%s-%s.fits' % (imobj.expnum, imobj.ccdname)
        print('Writing halo image to', fn)
        D.writeto(fn)
        # Append the halo image!
        import fitsio
        fitsio.write(fn, halo)
        fitsio.write(fn, image)
            
    return halo
