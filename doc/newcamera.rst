Adding a new telescope/camera
=============================

Outline
-------

The ``legacypipe`` data processing takes as inputs images that have been
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
``legacyzpts/legacy_zeropoints.py``.  Each camera has a special class
that handles all the specific details about how images from that
camera get treated.  The base class is ``legacypipe/image.py``.  This
includes functions that are used by the calibration & zeropoints
stage, as well as the legacypipe stage.

As an example, let's add the WIROPrime camera at the Wyoming InfraRed
Observatory (WIRO) telescope.  Our calibration & zeropoint call will
look something like::

    python -u legacyzpts/legacy_zeropoints.py --camera wiro --survey-dir wiro-dir --image wiro/20221003/a100_zbf.fit

Before we start, we'll need to set up a directory (``survey-dir``).
This directory will contain a subdirectory called ``images`` where all
your (reduced) images will be placed.  In the case of WIRO, there's
already a directory structure in place, so we'll instead use symlinks
to create the structure that ``legacypipe`` wants::

    mkdir -p wiro-dir/images
    ln -s /global/cfs/cdirs/desi/users/adamyers/wiro/reduced/ wiro-dir/images/wiro

so that, eg, the file ``wiro-dir/images/wiro/20221003/a100_zbf.fit`` exists.  (That is: the full file path is constructed as
the ``--survey-dir`` argument, plus ``images``, plus the path name you give.

There is a script that is meant to help you set up a new camera, called ``legacypipe/new-camera-setup.py``.  It takes the same
arguments as ``legacy_zeropoints.py``.  Let's call it and start following its instructions::

    > python legacypipe/new-camera-setup.py wiro/20221031/a144_zbf.fit  --camera wiro --survey-dir wiro-dir
    You must add your new camera to the list of known cameras at the top of the legacy_zeropoints.py script -- the CAMERAS variable.

There is a list of known cameras at the top of ``legacy_zeropoints.py``.  Add ``"wiro"`` there.

We then get::

    > python legacypipe/new-camera-setup.py wiro/20221031/a144_zbf.fit  --camera wiro --survey-dir wiro-dir
    You must:
     - create a new legacypipe.image.LegacySurveyImage subclass for your new camera
     - add it to the dict in legacypipe/survey.py : LegacySurveyData : self.image_typemap
     - import your new class in LegacySurveyData.__init__().

Have a look in ``legacypipe/survey.py``.  In the ``LegacySurveyData``
class there is a dictionary from camera names to their special classes
that handle data from that camera.  Let's add WIRO, pointing to a new
class that we will create.  Note also that earlier in that function we
import each of those special classes::

    from legacypipe.decam  import DecamImage
    from legacypipe.hsc    import HscImage
    from legacypipe.panstarrs import PanStarrsImage
    # Import our new class!
    from legacypipe.wiro import WiroImage

    self.image_typemap = {
      'decam'  : DecamImage,
      #...
      'hsc'    : HscImage,
      'panstarrs' : PanStarrsImage,
      # Add our new class to the dict!
      'wiro': WiroImage,
    }

We now need to create the new file ``legacypipe/wiro.py`` and create the
new class::

    from legacypipe.image import LegacySurveyImage

    class WiroImage(LegacySurveyImage):
        pass

Now the real work begins!  The ``legacy_zeropoints.py`` code will start
trying to read a bunch of headers out of your images.  The defaults
are based on what the CP does, so depending on exactly what headers
your instrument control system produces, you'll have to override a
number of functions from the ``LegacySurveyImage`` class.  For example,
for WIRO, the first error we get is

.. code-block:: bash

    > python legacypipe/new-camera-setup.py wiro/20221031/a144_zbf.fit --camera wiro --survey-dir wiro-dir
    For camera "wiro", found LegacySurveyImage subclass: <class 'legacypipe.wiro.WiroImage'>
    Reading wiro/20221031/a144_zbf.fit and trying to create new image object...
    Got image of type <class 'legacypipe.wiro.WiroImage'>
    Relative path to image file -- will be stored in the survey-ccds file --:  wiro/20221031/a144_zbf.fit
    Filesystem path to image file: wiro-dir/images/wiro/20221031/a144_zbf.fit
    Reading primary FITS header from image file...
    Reading a bunch of metadata from image primary header:
    get_band():
      -> "Filter"
    get_propid():
      -> ""
    get_expnum():
    Traceback (most recent call last):
      File "legacypipe/new-camera-setup.py", line 102, in <module>
        main()
      File "legacypipe/new-camera-setup.py", line 85, in main
        setattr(img, k, getattr(img, 'get_'+k)(primhdr))
      File "/global/homes/d/dstn/legacypipe/py/legacypipe/image.py", line 500, in get_expnum
        return primhdr['EXPNUM']
      File "/global/homes/d/dstn/fitsio2/fitsio/header.py", line 354, in __getitem__
        raise KeyError("unknown record: %s" % item)
    KeyError: 'unknown record: EXPNUM'

Have a look at the line following ``get_band():``: it's trying to read
the name of the filter from the header.  The WIRO images have a header
card like ``"FILTER = 'Filter 4: E 41102'"``, while the base-class
``image.py`` code returns only the first word.  We instead want to
return ``"NB_E"`` (narrow-band filter E) for this case, so we'll
override the ``get_band()`` function in our ``wiro.py`` class::

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

The next thing is ``get_propid()``: the name of the proposal.  WIRO
doesn't have this, and it's not essential, so we'll just leave it
blank.

Next is ``get_expnum()``.  This is an integer exposure number that is
used to uniquely identify the exposure.  WIRO doesn't have an exposure
number counter, so instead we'll cook one up out of the DATE header.
Add to the ``wiro.py`` code::

    def get_expnum(self, primhdr):
        date = primhdr['DATE-OBS']
        # DATE-OBS= '2022-10-04T05:20:19.335'
        d = datetime.strptime(date[:19], "%Y-%m-%dT%H:%M:%S")
        expnum = d.second + 100*(d.minute + 100*(d.hour + 100*(d.day + 100*(d.month + 100*d.year))))
        return expnum


Now, running it again and we get::

    get_band():
      -> "NB_E"
    get_propid():
      -> ""
    get_expnum():
      -> "20221101024745"
    get_camera():
      -> "wiroprime"
    get_exptime():
      -> "10.0"
    get_mjd():
      -> "None"
    get "HA" from primary header.
      -> "-59.80114"
    get "DATE" from primary header.
      -> "None"
    get "PLVER" from primary header.
      -> "None"
    get "PLPROCID" from primary header.
      -> "None"
    Will read image header from HDU 0
    Reading wiro-dir/images/wiro/20221031/a144_zbf.fit ext 0
    Reading image metadata...
    Got image size 4096 x 4096 pixels
    get_ccdname():
    Traceback (most recent call last):
      File "legacypipe/new-camera-setup.py", line 128, in <module>
        main()
      File "legacypipe/new-camera-setup.py", line 121, in main
        v = getattr(img, 'get_'+key)(primhdr, hdr)
      File "/global/homes/d/dstn/legacypipe/py/legacypipe/image.py", line 463, in get_ccdname
        return hdr['EXTNAME'].strip().upper()
      File "/global/homes/d/dstn/fitsio2/fitsio/header.py", line 354, in __getitem__
        raise KeyError("unknown record: %s" % item)
    KeyError: 'unknown record: EXTNAME'

The ``get_band()`` results looks good now, as does the
``get_expnum()``.  We have a few things to fix, though.  First,
``get_camera()`` should match the canonical camera name you've chosen,
ie, ``"wiro"`` in our case.  So we need to override the
``get_camera()`` function.  We probably want to have the correct MJD
for images (so we can correct for moving stars, etc), so we'll need to
parse the ``DATE-OBS`` header.  Finally, the error at the end: it's
trying to fetch the ``EXTNAME`` of our image extension, which for WIRO
doesn't exist because the WIRO image pixels are in the primary HDU.
So we'll fake that up too.  Adding to our ``WiroImage`` class::

    def get_mjd(self, primhdr):
        from astrometry.util.starutil_numpy import datetomjd
        d = self.get_date(primhdr)
        return datetomjd(d)

    def get_date(self, primhdr):
        date = primhdr['DATE-OBS']
        # DATE-OBS= '2022-10-04T05:20:19.335'
        d = datetime.strptime(date[:19], "%Y-%m-%dT%H:%M:%S")

    def get_camera(self, primhdr):
        cam = super().get_camera(primhdr)
        cam = {'wiroprime':'wiro'}.get(cam, cam)
        return cam

    def get_ccdname(self, primhdr, hdr):
        return 'CCD'

And now we get::

    ...
    get_camera():
      -> "wiro"
    get_exptime():
      -> "10.0"
    get_mjd():
      -> "59884.11649305555"
    get "HA" from primary header.
      -> "-59.80114"
    get "DATE" from primary header.
      -> "None"
    get "PLVER" from primary header.
      -> "None"
    get "PLPROCID" from primary header.
      -> "None"
    Will read image header from HDU 0
    Reading wiro-dir/images/wiro/20221031/a144_zbf.fit ext 0
    Reading image metadata...
    Got image size 4096 x 4096 pixels
    get_ccdname():
      -> "CCD"
    get_pixscale():
    Traceback (most recent call last):
      File "legacypipe/new-camera-setup.py", line 128, in <module>
        main()
      File "legacypipe/new-camera-setup.py", line 121, in main
        v = getattr(img, 'get_'+key)(primhdr, hdr)
      File "/global/homes/d/dstn/legacypipe/py/legacypipe/image.py", line 512, in get_pixscale
        return 3600. * np.sqrt(np.abs(hdr['CD1_1'] * hdr['CD2_2'] -
      File "/global/homes/d/dstn/fitsio2/fitsio/header.py", line 354, in __getitem__
        raise KeyError("unknown record: %s" % item)
    KeyError: 'unknown record: CD1_1'

By default, the code tries to figure out the pixel scale by reading
WCS header cards, but for WIRO these don't exist; we'll just return a
constant pixel scale::

    def get_pixscale(self, primhdr, hdr):
        return 0.58

For now, we'll ignore the ``get_fwhm()`` result, but we will need to fix it later.




        
Depending on your camera, you'll have to do this for a few different
header cards.

.......

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

Here, ``legacy_zeropoints.py`` is trying to read the data-quality (mask
bits) image.  And after that, it will try to read the inverse-variance
map image.  By default, it assumes these files are named the way the
CP names them.  You can change this by overriding the
``compute_filenames`` function from ``image.py``.  In the case of WIRO,
the masks and uncertainty maps are in HDUs following the image.  So we
can set the "dq" and "iv" filenames to be equal to the "image"
filename, but we're also going to have to set the HDU numbers, which
we have to do by overriding the ``__init__`` constructor for our class::

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

