class Duck(object):
    pass

def stage_largegalaxies(
        survey=None, targetwcs=None, bands=None, tims=None,
        brickname=None, version_header=None,
        plots=False, ps=None, coadd_bw=False, W=None, H=None,
        brick=None, blobs=None, lanczos=True, ccds=None,
        write_metrics=True,
        mp=None, record_event=None,
        **kwargs):

    import numpy as np
    from legacypipe.coadds import make_coadds, write_coadd_images
    from legacypipe.bits import DQ_BITS
    from legacypipe.survey import get_rgb, imsave_jpeg

    from tractor.image import Image
    from tractor.basics import NanoMaggies, LinearPhotoCal
    from tractor.sky import ConstantSky
    from tractor.wcs import ConstantFitsWcs
    from tractor import GaussianMixturePSF
    
    ## TODO -- custom sky subtraction!

    # Create coadds and then build custom tims from them.
    
    C = make_coadds(tims, bands, targetwcs,
                    detmaps=True, ngood=True, lanczos=lanczos,
                    allmasks=True, mp=mp, plots=plots, ps=ps,
                    callback=None,
                    #callback=write_coadd_images,
                    #callback_args=(survey, brickname, version_header, tims, targetwcs)
                    )
    with survey.write_output('image-jpeg', brick=brickname) as out:
        imsave_jpeg(out.fn, get_rgb(C.coimgs, bands), origin='lower')
    
    # TODO -- coadd PSF model?

    cotims = []
    for band,img,iv,mask in zip(bands, C.coimgs, C.cowimgs, C.allmasks):
        twcs = ConstantFitsWcs(targetwcs)
        # Totally bogus
        psf_sigma = np.mean([tim.psf_sigma for tim in tims if tim.band == band])

        psf = GaussianMixturePSF(1., 0., 0., psf_sigma**2, psf_sigma**2, 0.)
        cotim = Image(img, invvar=iv, wcs=twcs, psf=psf,
                      photocal=LinearPhotoCal(1., band=band),
                      sky=ConstantSky(0.), name='coadd-'+band)
        cotim.band = band
        cotim.subwcs = targetwcs
        cotim.psf_sigma = psf_sigma
        cotim.sig1 = 1./np.sqrt(np.median(iv[iv>0]))
        #cotim.dq = mask # hmm, not what we think this is
        cotim.dq = np.zeros(cotim.shape, dtype=np.int16)
        cotim.dq_saturation_bits = DQ_BITS['satur']
        cotim.psfnorm = 1./(2. * np.sqrt(np.pi) * psf_sigma)
        cotim.imobj = Duck()
        cotims.append(cotim)

        #import pdb ; pdb.set_trace()

        #return dict(cotims=cotims)
    # EVIL
    return dict(tims=cotims)

