import numpy as np

def subtract_halos(tims, refs, bands, mp, plots, ps):
    fluxes = np.zeros((len(refs),len(bands)))
    for i,b in enumerate(bands):
        mag = refs.get('decam_mag_%s' % b)
        fluxes[:,i] = 10.**((mag - 22.5) / -2.5)
    iband = dict([(b,i) for i,b in enumerate(bands)])
    args = [(tim, refs, fluxes[:,iband[tim.band]]) for tim in tims]
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

def moffat(rr, alpha, beta):
    return (beta-1.)/(np.pi * alpha**2)*(1. + (rr/alpha)**2)**(-beta)

def subtract_one_real(X):
    tim, refs, fluxes = X
    assert(np.all(refs.ref_epoch > 0))
    from legacypipe.survey import radec_at_mjd
    #print('Moving', len(refs), 'Gaia stars to MJD', tim.time.toMjd())
    rr,dd = radec_at_mjd(refs.ra, refs.dec, refs.ref_epoch.astype(float),
                         refs.pmra, refs.pmdec, refs.parallax, tim.time.toMjd())

    if tim.imobj.camera != 'decam':
        print('Warning: Stellar halo subtraction not implemented for cameras != Decam')
        return 0.

    halo = np.zeros(tim.shape, np.float32)
    for ref,flux,ra,dec in zip(refs, fluxes, rr, dd):
        H,W = tim.shape
        ok,x,y = tim.subwcs.radec2pixelxy(ra, dec)
        x -= 1.
        y -= 1.
        pixscale = tim.imobj.pixscale

        # Rongpu says only apply within 200"
        rad_arcsec = ref.radius * 3600.
        ### FIXME -- we're going to try subtracting the halo out to TWICE our masking radius.
        rad_arcsec *= 2.0
        rad_arcsec = np.minimum(rad_arcsec, 200.)
        pixrad = int(np.ceil(rad_arcsec / pixscale))

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

        # Inner apodization: ramp from 0 up to 1 between radii 6" and 6.8".
        # (Rongpu's "R3" and "R4")
        apr_i0 = 7. / pixscale
        apr_i1 = 8. / pixscale
        apodize *= np.clip((rads - apr_i0) / (apr_i1 - apr_i0), 0., 1.)

        if tim.band == 'z':
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
            if tim.imobj.ccdname.strip() in ['N20', 'S8', 'S10', 'S18', 'S21', 'S27']:
                alpha, beta, weight = 16, 2.3, 0.0095
            else:
                alpha, beta, weight = 17.650, 1.7, 0.0145

            # The 'pixscale**2' is because Rongpu's formula is in nanomaggies/arcsec^2
            halo[ylo:yhi+1, xlo:xhi+1] += (flux * apodize * weight *
                                           moffat(rads*pixscale, alpha, beta) * pixscale**2)

        else:
             fd = dict(g=0.00045,
                       r=0.00033)
             f = fd[tim.band]

             halo[ylo:yhi+1, xlo:xhi+1] += (flux * apodize * f * (rads*pixscale)**-2
                                            * pixscale**2)
        # We ASSUME tim is in nanomaggies units
    return halo

