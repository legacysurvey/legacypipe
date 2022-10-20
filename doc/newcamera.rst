Adding a new telescope/camera
=============================

Outline
-------

The `legacypipe` data processing takes as inputs images that have been
flat-fielded and had any other special instrument signatures removed.
It expects image pixels, a mask image (hot pixels, bad columns,
saturation, etc), and also an uncertainty image (including noise from
dark current, read noise, and sky background).  For the Legacy
Surveys, this instrument-signature-removal is done by different
versions of the (DECam) Community Pipeline by Frank Valdes, NOIRLab,
so often in the code these are referred to as "CP images".

The pipeline has these main steps:

  * calibration -- here, we compute, if necessary, astrometry, sky
    background, and PSF models.  For the Legacy Surveys, we use PSFEx
    for the PSF models, and a flexible spline model for the sky
    background ("splinesky").  The CP produces good astrometric
    headers (including distortion terms), but we do apply a simple
    offset to bring all images to Gaia-DR3.
  * zeropoints -- we compute astrometric and photometric zeropoints.
    Zeropointing uses the Pan-STARRS1 and Gaia-DR3 catalogs, and the
    calibration products.  We find catalog stars that are inside the
    image, and then use the Tractor code to perform a local fit that
    will refine the flux and RA,Dec position of each star.  Averages
    of those fluxes and positions relative to the reference catalogs
    yield astrometric and photometric zeropoints.  These zeropoints
    are recorded in FITS tables.
  * CCDs table -- after zeropoints have been computed, we gather up
    all the FITS tables and merge them into the "survey-CCDs" FITS
    table.  This file tells the pipeline about all the images that are
    available to be processed.
  * runbrick -- this is the main "legacypipe" or "tractor" processing.
    We process a 0.25 x 0.25-degree square of sky at once (a "brick"),
    reading in all the overlapping images found in the survey-CCDs
    file.  This code does source detection, source model fitting,
    forced-photometry of the WISE images, coadded image production,
    and catalog generation.


Calibrations
------------

Calibrations and zeropoints are performed by the script
`legacyzpts/legacy_zeropoints.py`.  Each camera has a special class
that handles all the specific details about how images from that
camera get treated.  The base class is `legacypipe/image.py`.  This
includes functions that are used by the calibration & zeropoints
stage, as well as the legacypipe stage.

As an example, let's add the WIROPrime camera at the Wyoming InfraRed
Observatory (WIRO) telescope.  Our calibration & zeropoint call will
look something like::

    python -u legacyzpts/legacy_zeropoints.py --camera wiro --survey-dir wiro-dir --image wiro/20221003/a100_zbf.fit

Before we start, we'll need to set up a directory (`survey-dir`).
This directory will contain a subdirectory called `images` where all
your (reduced) images will be placed.  In the case of WIRO, there's
already a directory structure in place, so we'll instead use symlinks
to create the structure that `legacypipe` wants::

    mkdir -p wiro-dir/images
    ln -s /global/cfs/cdirs/desi/users/adamyers/wiro/reduced/ wiro-dir/images/wiro

so that, eg, the file `wiro-dir/images/wiro/20221003/a100_zbf.fit` exists.

Now we can try running it and see what fails::

    > python -u legacyzpts/legacy_zeropoints.py --camera wiro --survey-dir wiro-dir --image wiro/20221003/a100_zbf.fit

If we run that, we'll get a complaint that the `wiro` camera isn't in the list of known cameras.  There is a list of known cameras at the top of `legacy_zeropoints.py`.  Add `"wiro"` there.  We then get a complaint::

    Traceback (most recent call last):
    File "legacyzpts/legacy_zeropoints.py", line 1541, in <module>
      main()
    File "legacyzpts/legacy_zeropoints.py", line 757, in main
      img = survey.get_image_object(None, camera=measureargs['camera'],
    File "/global/homes/d/dstn/legacypipe/py/legacypipe/survey.py", line 1625, in get_image_object
      imageType = self.image_class_for_camera(camera)
    File "/global/homes/d/dstn/legacypipe/py/legacypipe/survey.py", line 909, in image_class_for_camera
      return self.image_typemap[camera]
    KeyError: 'wiro'

Have a look in `legacypipe/survey.py`, where in the `LegacySurveyData`
class there is a dictionary from camera names to their special classes
that handle data from that camera.  Let's add WIRO, pointing to a new
class that we will create.  Note also that earlier in that function we
import each of those special classes::

    from legacypipe.decam  import DecamImage
    from legacypipe.hsc    import HscImage
    from legacypipe.panstarrs import PanStarrsImage
    # ...
    from legacypipe.wiro import WiroImage
    
    self.image_typemap = {
      'decam'  : DecamImage,
      #...
      'hsc'    : HscImage,
      'panstarrs' : PanStarrsImage,
      'wiro': WiroImage,
    }

We now need to create the new file `legacypipe/wiro.py` and create the
new class::

    from legacypipe.image import LegacySurveyImage
  
    class WiroImage(LegacySurveyImage):
        pass

Now the real work begins!  The `legacy_zeropoints.py` code will start
trying to read a bunch of headers out of your images.  The defaults
are based on what the CP does, so depending on exactly what headers
your instrument control system produces, you'll have to override a
number of functions from the `LegacySurveyImage` class.  For example,
for WIRO, the first error we get is::

    > python -u legacyzpts/legacy_zeropoints.py --camera wiro --survey-dir wiro-dir --image wiro/20221003/a100_zbf.fit

    ...
    Traceback (most recent call last):
    ...
    File "/global/homes/d/dstn/legacypipe/py/legacypipe/image.py", line 204, in __init__
      self.propid = self.get_propid(primhdr)
    File "/global/homes/d/dstn/legacypipe/py/legacypipe/image.py", line 469, in get_propid
      return primhdr['PROPID']
    File "/global/homes/d/dstn/fitsio2/fitsio/header.py", line 354, in __getitem__
      raise KeyError("unknown record: %s" % item)
    KeyError: 'unknown record: PROPID'

It's trying to get the proposal ID out of the header, but WIRO images
don't have this.  So we'll just return an empty string, by overriding
the relevant function in `legacypipe/image.py`::

    class WiroImage(LegacySurveyImage):
        def get_propid(self, primhdr):
            return ''

Depending on your camear, you'll have to do this for a few different
header cards.

Next, you are likely to get this complaint::

    Traceback (most recent call last):
    File "legacyzpts/legacy_zeropoints.py", line 1541, in <module>
      main()
    ...
    File "/global/homes/d/dstn/legacypipe/py/legacypipe/image.py", line 284, in __init__
      self.compute_filenames()
    File "/global/homes/d/dstn/legacypipe/py/legacypipe/image.py", line 344, in compute_filenames
      assert(self.dqfn != self.imgfn)
    AssertionError

Here, `legacy_zeropoints.py` is trying to read the data-quality (mask
bits) image.  And after that, it will try to read the inverse-variance
map image.  By default, it assumes these files are named the way the
CP names them.  You can change this by overriding the
`compute_filenames` function from `image.py`.  In the case of WIRO,
the masks and uncertainty maps are in HDUs following the image.  So we
can set the "dq" and "iv" filenames to be equal to the "image"
filename, but we're also going to have to set the HDU numbers, which
we have to do by overriding the `__init__` constructor for our class::

    class WiroImage(LegacySurveyImage):
    
        def __init__(self, survey, ccd, image_fn=None, image_hdu=0):
            super().__init__(survey, ccd, image_fn=image_fn, image_hdu=image_hdu)
            self.dq_hdu = 1
            self.wt_hdu = 2

        def compute_filenames(self):
            # Masks and weight-maps are in HDUs following the image
            self.dqfn = self.imgfn
            self.wtfn = self.imgfn

Okay, so now it should be reading from the correct file & HDUs.

Next, we get::

    > python -u legacyzpts/legacy_zeropoints.py --camera wiro --survey-dir wiro-dir --image wiro/20221003/a100_zbf.fit
    ...
    Working on image 1/1: wiro/20221003/a100_zbf.fit
    File not found /global/homes/d/dstn/wiro-data/reduced/20221003/a100_zbf.fit-annotated.fits
    File not found /global/homes/d/dstn/wiro-data/reduced/20221003/a100_zbf-psfex.fits
    File not found /global/homes/d/dstn/wiro-data/reduced/20221003/a100_zbf-splinesky.fits
    File not found /global/homes/d/dstn/wiro-data/reduced/20221003/a100_zbf.fit-photom.fits
    TIMING:before-run  Wall: 0.02 s, CPU: 0.01 s
    Got image object a100_zbf
    Traceback (most recent call last):
      File "legacyzpts/legacy_zeropoints.py", line 1541, in <module>
        main()
      File "legacyzpts/legacy_zeropoints.py", line 839, in main
        runit(F.imgfn, F.photomfn, F.annfn, mp, **measureargs)
      File "legacyzpts/legacy_zeropoints.py", line 534, in runit
        results = measure_image(imgfn, mp, survey=survey,
      File "legacyzpts/legacy_zeropoints.py", line 240, in measure_image
        assert(img.camera == camera)
    AssertionError

What's happening here is that the code is expecting to find the name
of the camera in the `INSTRUME` header keyword (converted to lower
case).  For the case of WIRO, the images have `INSTRUME=WIROPrime`,
but I decided to just call the camera `"wiro"`, so we'll have to sneak
in a fix for that::

    class WiroImage(LegacySurveyImage):
        # ...
        def get_camera(self, primhdr):
            cam = super().get_camera(primhdr)
            cam = {'wiroprime':'wiro'}.get(cam, cam)
            return cam
  
Now, re-running seems to get pretty far, but not much seems to happen::

    > python legacyzpts/legacy_zeropoints.py --camera wiro --survey-dir wiro-dir --image wiro/20221003/a100_zbf.fit    
    ...
    Working on image 1/1: wiro/20221003/a100_zbf.fit
    File not found wiro-dir/zpt/wiro/20221003/a100_zbf.fit-annotated.fits
    File not found wiro-dir/calib/psfex/wiro/20221003/a100_zbf-psfex.fits
    File not found wiro-dir/calib/sky/wiro/20221003/a100_zbf-splinesky.fits
    File not found wiro-dir/zpt/wiro/20221003/a100_zbf.fit-photom.fits
    TIMING:before-run  Wall: 0.02 s, CPU: 0.02 s
    Got image object a100_zbf
    TIMING:measure_image  Wall: 0.01 s, CPU: 0.01 s
    Wrote wiro-dir/zpt/wiro/20221003/a100_zbf.fit-annotated.fits
    TIMING:write-results-to-fits  Wall: 0.02 s, CPU: 0.02 s
    TIMING:after-run  Wall: 0.77 s, CPU: 0.76 s
    TIMING:total Wall: 0.79 s, CPU: 0.78 s

Have a look at that
`wiro-dir/zpt/wiro/20221003/a100_zbf.fit-annotated.fits` file -- it
contains zeros for all columns.  Turning on debugging messages with the `-v` flag, and deleting that `annotated` file to make it run again, we see the complaint::

    Got image object a100_zbf
    a100_zbf: Zero exposure time or low-level calibration flagged as bad; skipping image.

This is coming from another CP-specific check for whether the CP succeeded.  We can fix this by overriding the `calibration_good` function, making it always succeed::

    class WiroImage(LegacySurveyImage):
        # ...
        def calibration_good(self, primhdr):
            return True

Next, we're going to have to tell the pipeline which FITS image
extensions it should use -- many multi-CCD cameras produce FITS files
with one image per HDU, identified with the `EXTNAME` header card.
The WIROPrime camera just has one chip, and the primary HDU has no
EXTNAME, so we can tell the pipeline to read the HDU index instead.
We'll also fake up a CCD name::

    class WiroImage(LegacySurveyImage):
        # ...
        def get_extension_list(self, debug=False):
            return [0]
        def get_ccdname(self, primhdr, hdr):
            return 'CCD'

The pipeline is now trying to produce calibration products for the
image.  For WIRO, there are no astrometric headers at all, so we'll
have to perform the astrometric calibration ourselves.  We'll also
need to tell the pipeline the pixel scale of the camera::

    class WiroImage(LegacySurveyImage):
        # ...
        def get_pixscale(self, primhdr, hdr):
            # arcsec/pixel
            return 0.58

Next, the pipeline will try to run SourceExtractor and PsfEx to
generate PSF models.  It will search for config files like
`legacypipe/py/legacypipe/config/wiro.param`, but for most of the
cameras we handle, a common config files is used, so this can usually
be handled with a symlink of `wiro.param` to `common.param`,
`wiro.psfex` to `common.psfex`, `wiro.se` to `common.se`, and
`gauss_5.0_9x9.conv` to `wiro.conv`.



  

  

        
Zeropoints
----------


Legacypipe/the Tractor
----------------------

