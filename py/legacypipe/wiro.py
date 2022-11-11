from datetime import datetime

from legacypipe.image import LegacySurveyImage

import logging
logger = logging.getLogger('legacypipe.wiro')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

class WiroImage(LegacySurveyImage):
    def __init__(self, survey, ccd, image_fn=None, image_hdu=0, **kwargs):
        super().__init__(survey, ccd, image_fn=image_fn, image_hdu=image_hdu, **kwargs)
#         self.dq_hdu = 1
#         self.wt_hdu = 2

    def get_band(self, primhdr):
        f = primhdr['FILTER']
        filtmap = {
            'Filter 1: g 1736'  : 'g',
            'Filter 2: C 14859' : 'NB_C',
            'Filter 3: D 27981' : 'NB_D',
            'Filter 4: E 41102' : 'NB_E',
            'Filter 5: A 54195' : 'NB_A',
        }
        # ASSUME that the filter is one of the above!
        return filtmap[f]

    def get_expnum(self, primhdr):
        d = self.get_date(primhdr)
        expnum = d.second + 100*(d.minute + 100*(d.hour + 100*(d.day + 100*(d.month + 100*d.year))))
        return expnum

    def get_mjd(self, primhdr):
        from astrometry.util.starutil_numpy import datetomjd
        d = self.get_date(primhdr)
        return datetomjd(d)

    def get_date(self, primhdr):
        date = primhdr['DATE-OBS']
        # DATE-OBS= '2022-10-04T05:20:19.335'
        return datetime.strptime(date[:19], "%Y-%m-%dT%H:%M:%S")

    def get_camera(self, primhdr):
        cam = super().get_camera(primhdr)
        cam = {'wiroprime':'wiro'}.get(cam, cam)
        return cam

    def get_ccdname(self, primhdr, hdr):
        return 'CCD'

    def get_pixscale(self, primhdr, hdr):
        return 0.58

#     def compute_filenames(self):
#         # Masks and weight-maps are in HDUs following the image
#         self.dqfn = self.imgfn
#         self.wtfn = self.imgfn
#     def calibration_good(self, primhdr):
#         return True
#     def get_extension_list(self, debug=False):
#         return [0]

    #def funpack_files(self, imgfn, maskfn, imghdu, maskhdu, todelete):
    
