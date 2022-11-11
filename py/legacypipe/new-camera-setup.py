import os
import sys
import logging

from legacypipe.survey import LegacySurveyData

logger = logging.getLogger('legacypipe.new-camera-setup')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)


def main():
    from legacyzpts.legacy_zeropoints import CAMERAS

    
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--camera', required=True)

    parser.add_argument('--image-hdu', default=0, type=int, help='Read image data from the given HDU number')

    parser.add_argument('--survey-dir', type=str, default=None,
                        help='Override the $LEGACY_SURVEY_DIR environment variable')
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='More logging')

    parser.add_argument('image', metavar='image-filename', help='Image filename to read')

    opt = parser.parse_args()

    if opt.verbose:
        lvl = logging.DEBUG
    else:
        lvl = logging.INFO
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    # tractor logging is *soooo* chatty
    logging.getLogger('tractor.engine').setLevel(lvl + 10)

    if not opt.camera in CAMERAS:
        print('You must add your new camera to the list of known cameras at the top of the legacy_zeropoints.py script -- the CAMERAS variable.')
        return

    survey = LegacySurveyData(survey_dir=opt.survey_dir)

    clazz = None
    try:
        clazz = survey.image_class_for_camera(opt.camera)
    except KeyError:
        print('You must:')
        print(' - create a new legacypipe.image.LegacySurveyImage subclass for your new camera')
        print(' - add it to the dict in legacypipe/survey.py : LegacySurveyData : self.image_typemap')
        print(' - import your new class in LegacySurveyData.__init__().')
        return

    info('For camera "%s", found LegacySurveyImage subclass: %s' % (opt.camera, str(clazz)))
    
    info('Reading', opt.image, 'and trying to create new image object...')

    img = survey.get_image_object(None, camera=opt.camera,
                                  image_fn=opt.image, image_hdu=opt.image_hdu,
                                  camera_setup=True)
                                  
    info('Got image of type', type(img))

    # Here we're copying some code out of image.py...
    image_fn = opt.image
    image_hdu = opt.image_hdu
    img.image_filename = image_fn
    img.imgfn = os.path.join(img.survey.get_image_dir(), image_fn)

    info('Relative path to image file -- will be stored in the survey-ccds file --: ', img.image_filename)
    info('Filesystem path to image file:', img.imgfn)

    if not os.path.exists(img.imgfn):
        print('Filesystem path does not exist.  Should be survey-dir path + images (%s) + image-file-argument (%s)' % (survey.get_image_dir(), image_fn))
        return

    info('Reading primary FITS header from image file...')
    primhdr = img.read_image_primary_header()

    info('Reading a bunch of metadata from image primary header:')

    for k in ['band', 'propid', 'expnum', 'camera', 'exptime']:
        info('get_%s():' % k)
        v = getattr(img, 'get_'+k)(primhdr)
        info('  -> "%s"' % v)
        setattr(img, k, v)

    info('get_mjd():')
    img.mjdobs = img.get_mjd(primhdr)
    info('  -> "%s"' % img.mjdobs)

    namechange = {'date': 'procdate',}
    for key in ['HA', 'DATE', 'PLVER', 'PLPROCID']:
        info('get "%s" from primary header.' % key)
        val = primhdr.get(key)
        if isinstance(val, str):
            val = val.strip()
            if len(val) == 0:
                raise ValueError('Empty header card: %s' % key)
        key = namechange.get(key.lower(), key.lower())
        key = key.replace('-', '_')
        info('  -> "%s"' % val)
        setattr(img, key, val)

    img.hdu = image_hdu
    info('Will read image header from HDU', image_hdu)
    hdr = img.read_image_header(ext=image_hdu)
    info('Reading image metadata...')
    hinfo = img.read_image_fits()[image_hdu].get_info()
    #info('Got:', hinfo)
    img.height,img.width = hinfo['dims']
    info('Got image size', img.width, 'x', img.height, 'pixels')
    img.hdu = hinfo['hdunum'] - 1
    for key in ['ccdname', 'pixscale', 'fwhm']:
        info('get_%s():' % key)
        v = getattr(img, 'get_'+key)(primhdr, hdr)
        info('  -> "%s"' % v)
        setattr(img, key, v)



if __name__ == '__main__':
    main()
