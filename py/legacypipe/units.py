def get_units_for_columns(cols, bands=None, extras=None):
    if bands is None:
        bands = []
    deg = 'deg'
    degiv = 'deg^(-2)'
    arcsec = 'arcsec'
    arcseciv = 'arcsec^(-2)'
    flux = 'nanomaggies'
    fluxiv = 'nanomaggies^(-2)'
    pm = 'mas/yr'
    pmiv = '(mas/yr)^(-2)'
    unitmap = dict(
        ra=deg, dec=deg, ra_ivar=degiv, dec_ivar=degiv,
        ebv='mag',
        shape_r=arcsec,
        shape_r_ivar=arcseciv)
    unitmap.update(pmra=pm, pmdec=pm, pmra_ivar=pmiv, pmdec_ivar=pmiv,
                 parallax='mas', parallax_ivar='mas^(-2)')
    unitmap.update(gaia_phot_g_mean_mag='mag',
                 gaia_phot_bp_mean_mag='mag',
                 gaia_phot_rp_mean_mag='mag')
    # units used in forced phot.
    unitmap.update(exptime='sec',
                   flux=flux, flux_ivar=fluxiv,
                   apflux=flux, apflux_ivar=fluxiv,
                   psfdepth=fluxiv, galdepth=fluxiv,
                   sky='nanomaggies/arcsec^2',
                   psfsize=arcsec,
                   fwhm='pixels',
                   ccdrarms=arcsec, ccddecrms=arcsec,
                   skyrms='counts/sec',
                   dra=arcsec, ddec=arcsec,
                   dra_ivar=arcseciv, ddec_ivar=arcseciv)
    # Fields that have band suffixes
    funits = dict(
        flux=flux, flux_ivar=fluxiv,
        apflux=flux, apflux_ivar=fluxiv, apflux_resid=flux,
        apflux_blobresid=flux,
        psfdepth=fluxiv, galdepth=fluxiv, psfsize=arcsec,
        fiberflux=flux, fibertotflux=flux,
        lc_flux=flux, lc_flux_ivar=fluxiv,
    )
    for b in bands:
        unitmap.update([('%s_%s' % (k, b), v)
                      for k,v in funits.items()])

    if extras is not None:
        unitmap.update(extras)

    # Create a list of units aligned with 'cols'
    units = [unitmap.get(c, '') for c in cols]
    return units
