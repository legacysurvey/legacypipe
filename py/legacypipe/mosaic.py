from __future__ import print_function

from legacypipe.image import LegacySurveyImage

class MosaicImage(LegacySurveyImage):
    '''
    Class for handling images from the Mosaic3 camera processed by the
    NOAO Community Pipeline.
    '''

    mjd_third_pixel_fixed = 57674.

    def __init__(self, survey, t):
        super(MosaicImage, self).__init__(survey, t)

        for attr in ['imgfn', 'dqfn', 'wtfn']:
            fn = getattr(self, attr)
            if os.path.exists(fn):
                continue
            # Workaround: exposure numbers 330667 through 330890 at
            # least have some of the files named "v1" and some named
            # "v2".  Try both.
            if 'v1' in fn:
                fnother = fn.replace('v1', 'v2')
                if os.path.exists(fnother):
                    print('Using', fnother, 'rather than', fn)
                    setattr(self, attr, fnother)
                    fn = fnother
            elif 'v2' in fn:
                fnother = fn.replace('v2', 'v1')
                if os.path.exists(fnother):
                    print('Using', fnother, 'rather than', fn)
                    setattr(self, attr, fnother)
                    fn = fnother

    def remap_invvar(self, invvar, primhdr, img, dq):
        return self.remap_invvar_shotnoise(invvar, primhdr, img, dq)

