import numpy as np

def subtract_halos(tims, refs, bands, mp, plots, ps):
    fluxes = np.zeros((len(refs),len(bands)))
    color = refs.phot_bp_mean_mag - refs.phot_rp_mean_mag
    color[np.logical_not(np.isfinite(color))] = 0.
    color = np.clip(color, -0.5, 3.3)
    G = refs.phot_g_mean_mag
    for i,b in enumerate(bands):
        # Use Arjun's Gaia-to-DECam transformations.
        coeffs = dict(
            g=[-0.11368, 0.37504, 0.17344, -0.08107, 0.28088,
               -0.21250, 0.05773,-0.00525],
            r=[ 0.10533,-0.22975, 0.06257,-0.24142, 0.24441,
                -0.07248, 0.00676],
            z=[ 0.46744,-0.95143, 0.19729,-0.08810, 0.01566])[b]
        mag = G
        for order,c in enumerate(coeffs):
            mag += c * color**order
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

def subtract_one_real(X):
    tim, refs, fluxes = X

    assert(np.all(refs.ref_epoch > 0))
    from legacypipe.survey import radec_at_mjd
    print('Moving', len(refs), 'Gaia stars to MJD', tim.time.toMjd())
    rr,dd = radec_at_mjd(refs.ra, refs.dec, refs.ref_epoch.astype(float),
                         refs.pmra, refs.pmdec, refs.parallax, tim.time.toMjd())

    halo = np.zeros(tim.shape, np.float32)
    for ref,flux,ra,dec in zip(refs, fluxes, rr, dd):
        H,W = tim.shape
        ok,x,y = tim.subwcs.radec2pixelxy(ra, dec)
        x -= 1.
        y -= 1.
        pixscale = tim.imobj.pixscale
        pixrad = int(np.ceil(ref.radius * 3600. / pixscale))

        ### FIXME -- we're going to try subtracting the halo out to TWICE our masking radius.
        pixrad *= 2.

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
        apr_i0 = 6.0 / pixscale
        apr_i1 = 6.8 / pixscale
        apodize *= np.clip((rads - r0) / (r1 - r0), 0., 1.)

        # The analytic profiles are:
        #  f = 10^(c5 * x^5 + c4 * x^4 + ... + c1 * x^1 + c0) 
        # where x = log10(radius); the radius is in arcsec; the surface brightness f is normalized to a star of magnitude 22.5
        # The coefficients for the three bands are (in the order of c5, c4, ..., c0): 
        coeffs = dict(
            g = [-0.25374275, 0.2667515, 1.4606936, -1.99823195, -3.39780119, -2.23553798],
            r = [-0.02682569, -0.4871116, 2.31043612, -2.60471433, -2.96514646, -2.39183566],
            z = [1.4393356, -7.96924951, 16.0064581, -13.63062087, 1.18527603, -2.86520728])[tim.band]

        # Reverse the coeffs
        coeffs = list(reversed(coeffs))

        # Convert "rads" to log_10(arcsec)
        xx = np.log10(rads * pixscale)

        hh = np.zeros(rads.shape)
        for order,c in enumerate(coeffs):
            hh += c * xx**order

        hh = flux * 10.**hh

        # ASSUME tim is in nanomaggies units
        halo[ylo:yhi+1, xlo:xhi+1] += hh
    return halo

