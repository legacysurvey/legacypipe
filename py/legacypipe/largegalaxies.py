from legacypipe.coadds import make_coadds
from legacypipe.bits import DQ_BITS



def stage_largegalaxies(
        survey=None, targetwcs=None, bands=None, tims=None,
        brickname=None, version_header=None,
        plots=False, ps=None, coadd_bw=False, W=None, H=None,
        brick=None, blobs=None, lanczos=True, ccds=None,
        write_metrics=True,
        mp=None, record_event=None,
        **kwargs):

    ## TODO -- custom sky subtraction!

    # Create coadds and then build custom tims from them.
    
    C = make_coadds(tims, bands, targetwcs,
                    detmaps=True, ngood=True, lanczos=lanczos,
                    callback=None,
                    mp=mp, plots=plots, ps=ps)
    #write_coadd_images,
    #callback_args=(survey, brickname, version_header, tims,
    #targetwcs),

    from tractor.image import Image
    from tractor.basics import NanoMaggies, LinearPhotoCal
    from tractor.sky import ConstantSky
    from tractor.wcs import ConstantFitsWcs
    from tractor import GaussianMixturePSF
    
    # TODO -- coadd PSF model?

    class Duck(object):
        pass

    cotims = []
    for band,img,iv,mask in zip(bands, C.coimgs, C.cowimgs, C.andmask):
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
        cotim.dq = mask
        cotim.dq_saturation_bits = DQ_BITS['satur']
        cotim.psfnorm = 1./(2. * np.sqrt(np.pi) * psf_sigma)
        cotim.imobj = Duck()
        cotims.append(cotim)

        #return dict(cotims=cotims)
    # EVIL
    return dict(tims=cotims)

