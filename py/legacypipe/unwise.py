import numpy as np

def unwise_phot(X):
    from wise.forcedphot import unwise_forcedphot
    (wcat, tiles, band, roiradec, unwise_dir, wise_ceres, broadening,
    pixelized_psf, get_mods) = X
    kwargs = dict(roiradecbox=roiradec, bands=[band],
                  unwise_dir=unwise_dir, psf_broadening=broadening,
                  pixelized_psf=pixelized_psf)
    if get_mods:
        # This requires a newer version of the tractor code!
        kwargs.update(get_models=get_mods)

    ### FIXME
    #kwargs.update(save_fits=True)

    try:
        W = unwise_forcedphot(wcat, tiles, use_ceres=wise_ceres, **kwargs)
    except:
        import traceback
        print('unwise_forcedphot failed:')
        traceback.print_exc()

        if wise_ceres:
            print('Trying without Ceres...')
            try:
                W = unwise_forcedphot(wcat, tiles, use_ceres=False, **kwargs)
            except:
                print('unwise_forcedphot failed (2):')
                traceback.print_exc()
                if get_mods:
                    print('--unwise-coadds requires a newer tractor version...')
                    kwargs.update(get_models=False)
                    W = unwise_forcedphot(wcat, tiles, use_ceres=False, **kwargs)
                    W = W,dict()
        else:
            W = None
    return W

def collapse_unwise_bitmask(bitmask, band):
    '''
    Converts WISE mask bits (in the unWISE data products) into the
    more compact codes reported in the tractor files as
    WISEMASK_W[12], and the "maskbits" WISE extensions.

    output bits :
    # 2^0 = bright star core and wings
    # 2^1 = PSF-based diffraction spike
    # 2^2 = optical ghost
    # 2^3 = first latent
    # 2^4 = second latent
    # 2^5 = AllWISE-like circular halo
    # 2^6 = bright star saturation
    # 2^7 = geometric diffraction spike
    '''
    assert((band == 1) or (band == 2))
    from collections import OrderedDict

    bits_w1 = OrderedDict([('core_wings', 2**0 + 2**1),
                           ('psf_spike', 2**27),
                           ('ghost', 2**25 + 2**26),
                           ('first_latent', 2**13 + 2**14),
                           ('second_latent', 2**17 + 2**18),
                           ('circular_halo', 2**23),
                           ('saturation', 2**4),
                           ('geom_spike', 2**29)])

    bits_w2 = OrderedDict([('core_wings', 2**2 + 2**3),
                           ('psf_spike', 2**28),
                           ('ghost', 2**11 + 2**12),
                           ('first_latent', 2**15 + 2**16),
                           ('second_latent', 2**19 + 2**20),
                           ('circular_halo', 2**24),
                           ('saturation', 2**5),
                           ('geom_spike', 2**30)])

    bits = (bits_w1 if (band == 1) else bits_w2)

    # hack to handle both scalar and array inputs
    result = 0*bitmask
    for i, feat in enumerate(bits.keys()):
        result += ((2**i)*(np.bitwise_and(bitmask, bits[feat]) != 0)).astype(np.uint8)
    return result.astype('uint8')

