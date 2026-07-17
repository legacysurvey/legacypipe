import numpy as np

from legacypipe.decam import DecamImage
from legacypipe.survey import LegacySurveyData

'''
Testing code for adding noise to deeper images to simulate DECaLS depth data.
'''

class DecamImagePlusNoise(DecamImage):
    '''
    A DecamImage subclass to add noise to DECam images upon read.
    '''
    def __init__(self, survey, t):
        t.camera = 'decam'
        super(DecamImagePlusNoise, self).__init__(survey, t)
        self.addnoise = t.addnoise

    def get_tractor_image(self, **kwargs):
        assert(kwargs.get('nanomaggies', True))
        tim = super(DecamImagePlusNoise, self).get_tractor_image(**kwargs)
        if tim is None:
            return None
        with np.errstate(divide='ignore'):
            ie = 1. / (np.hypot(1. / tim.inverr, self.addnoise))
        ie[tim.inverr == 0] = 0.
        tim.setInvError(ie)
        tim.data += np.random.normal(size=tim.shape) * self.addnoise
        print('Adding noise: sig1 was', tim.sig1)
        print('Adding', self.addnoise)
        sig1 = 1. / np.median(ie[ie > 0])
        print('New sig1 is', sig1)
        tim.sig1 = sig1
        return tim

class SubsetSurvey(LegacySurveyData):
    def __init__(self, subset=0, fill_to_depth=None, **kwargs):
        super(SubsetSurvey, self).__init__(**kwargs)
        self.subset = subset
        if fill_to_depth is None:
            self.fill_to_detiv = None
        else:
            from tractor import NanoMaggies
            Nsigma = 5.
            sig = NanoMaggies.magToNanomaggies(fill_to_depth) / Nsigma
            targetiv = 1./sig**2
            self.fill_to_detiv = targetiv
        self.image_typemap.update({'decam+noise' : DecamImagePlusNoise})

    def get_ccds(self, **kwargs):
        print('SubsetSurvey.get_ccds()')
        CCDs = super(SubsetSurvey, self).get_ccds(**kwargs)
        print('SubsetSurvey: got', len(CCDs), 'CCDs')
        if self.fill_to_detiv is None:
            CCDs.cut(CCDs.subset == self.subset)
            print('After cutting to subset %i: %i CCDs' %
                  (self.subset, np.sum(CCDs.subset == self.subset)))
        else:
            CCDs.cut(np.append(np.flatnonzero(CCDs.subset == self.subset),
                               np.flatnonzero(CCDs.subset == -1)))
            print('After cutting to subset %i: %i CCDs in set + %i filler' %
                (self.subset, np.sum(CCDs.subset == self.subset), np.sum(CCDs.subset == -1)))
        return CCDs

    def filter_tims(self, tims, ccds, targetwcs)
        if self.fill_to_detiv is None:
            return tims, ccds
        from astrometry.util.resample import resample_with_wcs, OverlapError
        keep_tims = []
        keep_i = []
        bands = set([tim.band for tim in tims])
        for band in bands:
            h,w = targetwcs.shape
            detiv = np.zeros((h,w), np.float32)
            for itim,tim in enumerate(tims):
                if tim.band != band:
                    continue
                Yo = None
                try:
                    Yo,Xo,Yi,Xi,_ = resample_with_wcs(targetwcs, tim.subwcs)
                except OverlapError:
                    pass
                psfnorm = ccds.psfnorm[itim]
                # sig1 or tim's invvar??
                # for images where the invvar takes the source counts into effect, we
                # don't want to try to fill in around bright objects
                # or fill small holes like CRs or satellite trails, probably
                # apodization??
                tim_detiv = 1. / (tim.sig1 / psfnorm)**2
                if ccds.subset[itim] != -1:
                    detiv[Yo,Xo] += tim_detiv
                    keep_i.append(itim)
                    keep_tims.append(tim)
                else:
                    keep_pix = (detiv[Yo,Xo] < self.fill_to_detiv)
                    if not np.any(keep_pix):
                        continue
                    Yo,Xo,Yi,Xi = Yo[keep_pix], Xo[keep_pix], Yi[keep_pix], Xi[keep_pix]
                    # Compute how much noise should be added to hit the target depth (if any)
                    targetdetiv = self.fill_to_detiv - detiv[Yo,Xo]
                    # detiv = psfnorm^2 / sig1^2
                    targetiv = psfnorm**2 / targetdetiv
                    addnoise = np.sqrt(np.maximum(0., 1. / targetiv - tim.sig1**2))
                    tim.data[Yi,Xi] += addnoise * np.random.normal(size=len(Yi))
                    newie = 1./np.hypot(1./tim.inverr[Yi,Xi], addnoise)
                    newie[tim.inverr[Yi,Xi] == 0] = 0
                    # zero out everything except for the pixels we're keeping
                    tim.inverr[:,:] = 0.
                    tim.inverr[Yi,Xi] = newie
                    tim.sig1 = 1./np.median(tim.inverr[tim.inverr != 0])
                    y0 = np.min(Yi)
                    y1 = np.max(Yi+1)
                    x0 = np.min(Xi)
                    x1 = np.max(Xi+1)
                    slc = slice(y0,y1), slice(x0,x1)
                    tim.data = tim.data[slc]
                    tim.inverr = tim.inverr[slc]
                    if tim.dq is not None:
                        tim.dq = tim.dq[slc]
                    tim.wcs = tim.wcs.shifted(x0, y0)
                    tim.psf = tim.psf.getShifted(x0, y0)
                    tim.sky = tim.sky.shifted(x0, y0)
                    tim.subwcs = tim.subwcs.get_subimage(x0, y0, x1-x0, y1-y0)
                    detiv[Yo,Xo] += psfnorm**2 * tim.inverr[Yi,Xi]**2
                    keep_i.append(itim)
                    keep_tims.append(tim)

        return keep_tims, ccds.cut(np.array(keep_i))

def main():
    from runbrick import run_brick, get_parser, get_runbrick_kwargs

    parser = get_parser()
    # subset number
    parser.add_argument('--subset', type=int, help='COSMOS subset number [0 to 4, 10 to 12]', default=0)
    # ugh, here we're assuming this is constant for all bands....
    parser.add_argument('--fill-to-depth', type=float, default=None
                        help='Fill in images with subset==-1 to reach given target depth in mags')
    opt = parser.parse_args()
    if opt.brick is None and opt.radec is None:
        parser.print_help()
        return -1

    optdict = vars(opt)
    verbose = optdict.pop('verbose')
    subset = optdict.pop('subset')
    survey, kwargs = get_runbrick_kwargs(**optdict)
    if kwargs in [-1,0]:
        return kwargs

    import logging
    if verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    bands = opt.bands
    if bands is None:
        bands = ['g','r','z']
    else:
        bands = bands.split(',')
    survey = SubsetSurvey(survey_dir=opt.survey_dir, subset=subset,
                          output_dir=opt.output_dir,
                          allbands=bands,
                          fill_to_depth=opt.fill_to_depth)
    print('Using survey:', survey)
    #print('with output', survey.output_dir)

    print('opt:', opt)
    print('kwargs:', kwargs)

    run_brick(opt.brick, survey, **kwargs)
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
