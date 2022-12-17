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
        return ''

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
      -> ""
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

Depending on your camera, you'll have to override various other metadata-fetching functions.

For now, we'll ignore the ``get_fwhm()`` result, but we will need to fix it later.

Re-running, we get::

    ....
    get_pixscale():
      -> "0.58"
    get_fwhm():
      -> "nan"
    Traceback (most recent call last):
      File "legacypipe/new-camera-setup.py", line 246, in <module>
        main()
      File "legacypipe/new-camera-setup.py", line 130, in main
        img.compute_filenames()
      File "/global/homes/d/dstn/legacypipe/py/legacypipe/image.py", line 237, in compute_filenames
        assert(self.dqfn != self.imgfn)
    AssertionError

Here, the code is trying to read the data-quality (mask bitmask)
image.  And after that, it will try to read the inverse-variance map
image.  By default, it assumes these files are named the way the CP
names them.  You can change this by overriding the
``compute_filenames`` function from ``image.py``.  In the case of
WIRO, the masks and uncertainty maps are in HDUs following the image.
So we can set the "dq" and "iv" filenames to be equal to the "image"
filename, but we're also going to have to set the HDU numbers, which
we have to do by overriding the ``__init__`` constructor for our
class::

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

Now we get::

    ...
    get_fwhm():
    -> "nan"
    Will read image pixels from file         wiro-dir/images/wiro/20221031/a144_zbf.fit HDU 0
    Will read inverse-variance map from file wiro-dir/images/wiro/20221031/a144_zbf.fit HDU 2
    Will read data-quality map from file     wiro-dir/images/wiro/20221031/a144_zbf.fit HDU 1
    Will read images from these FITS HDUs: ['MASK', 'UNCERT']
    Source Extractor & PsfEx will read the following config files:
      SE config: /global/homes/d/dstn/legacypipe/py/legacypipe/config/wiro.se (does not exist)
      SE params: /global/homes/d/dstn/legacypipe/py/legacypipe/config/wiro.param (does not exist)
      SE convolution filter: /global/homes/d/dstn/legacypipe/py/legacypipe/config/wiro.conv (does not exist)
      PsfEx config: /global/homes/d/dstn/legacypipe/py/legacypipe/config/wiro.psfex (does not exist)
    could not find special psfex configuration file for wiro not using per-image psfex configurations.
    Special PsfEx flags for this CCD: 
    RA,Dec boresight: 35.97804166666666 40.68613888888889
    Airmass: 1.4
    Traceback (most recent call last):
      File "legacypipe/new-camera-setup.py", line 246, in <module>
        main()
      File "legacypipe/new-camera-setup.py", line 156, in main
        info('Gain:', img.get_gain(primhdr, hdr))
      File "/global/homes/d/dstn/legacypipe/py/legacypipe/image.py", line 352, in get_gain
        return primhdr['GAIN']
      File "/global/homes/d/dstn/fitsio2/fitsio/header.py", line 354, in __getitem__
        raise KeyError("unknown record: %s" % item)
    KeyError: 'unknown record: GAIN'

We have a couple of issues to fix.  First, the list of HDUs.  By
default, ``image.py`` will try to read all HDUs in the file as CCD
images, assuming a multi-CCD camera with images stored as multi-HDU
FITS files.  For WIRO, there is only a single CCD, so we must override
the ``get_extension_list`` function.  Later, we also see that a
``GAIN`` header card is expected::

    class WiroImage(LegacySurveyImage):
        # ...
        def get_gain(self, primhdr, hdr):
            # from https://iopscience.iop.org/article/10.1088/1538-3873/128/969/115003/ampdf
            return 2.6
    
        def get_extension_list(self, debug=False):
            return [0]
  
In addition, we see that the expected Source Extractor and PsfEx
configuration file do not exist.  Config files are expected to be found in, eg,
``legacypipe/py/legacypipe/config/wiro.param``, but for most of the
cameras we handle, a common config files is used, so this can usually
be handled with a symlink of ``wiro.param`` to ``common.param``,
``wiro.psfex`` to ``common.psfex``, ``wiro.se`` to ``common.se``, and
``gauss_5.0_9x9.conv`` to ``wiro.conv``.

Doing all this, we get::

    > python legacypipe/new-camera-setup.py wiro/20221031/a144_zbf.fit --camera wiro --survey-dir wiro-dir
    ...
    Gain: 2.6
    Traceback (most recent call last):
      File "legacypipe/new-camera-setup.py", line 246, in <module>
        main()
      File "legacypipe/new-camera-setup.py", line 157, in main
        info('WCS Reference pixel CRPIX[12]:', hdr['CRPIX1'], hdr['CRPIX2'])
      File "/global/homes/d/dstn/fitsio2/fitsio/header.py", line 354, in __getitem__
        raise KeyError("unknown record: %s" % item)
    KeyError: 'unknown record: CRPIX1'

Here, the code is looking for an astrometric header in the image.
WIRO lacks astrometric headers entirely, so at this point we're going
to have to do some real work!

It turns out that we can use Astrometry.net to generate WCS headers
for the WIRO images, so when asked to return the WCS for an image, we
will have to read the WCS header from an external file, which we'll
link to in the ``calib/`` directory.

We add the following to the ``WiroImage`` class::

    class WiroImage(LegacySurveyImage):
        # ...
        def get_wcs(self, hdr=None):
            calibdir = self.survey.get_calib_dir()
            imgdir = os.path.dirname(self.image_filename)
            fn = os.path.join(calibdir, 'wcs', imgdir, self.name + '.wcs')
            from astrometry.util.util import Sip
            return Sip(fn)
    
        def get_crpixcrval(self, primhdr, hdr):
            wcs = self.get_wcs()
            p1,p2 = wcs.get_crpix()
            v1,v2 = wcs.get_crval()
            return p1,p2,v1,v2
    
        def get_cd_matrix(self, primhdr, hdr):
            wcs = self.get_wcs()
            return wcs.get_cd()
    
With this, the code gets much further::

    > python -u legacypipe/new-camera-setup.py --camera wiro --survey-dir wiro-dir wiro/20221031/a144_zbf.fit
    ...
    Gain: 2.6
    WCS filename: wiro-dir/calib/wcs/wiro/20221031/a144_zbf.wcs
    WCS Reference pixel: 2048.5 2048.5
    WCS Reference pos: 35.0501564555 40.0328664229
    WCS filename: wiro-dir/calib/wcs/wiro/20221031/a144_zbf.wcs
    WCS CD matrix: (-1.44356992147e-06, -0.000159826959704, -0.000159689121508, 1.35262350325e-06)
    WCS filename: wiro-dir/calib/wcs/wiro/20221031/a144_zbf.wcs
    Got WCS object: SIP(TAN): crpix (2048.5, 2048.5), crval (35.0502, 40.0329), cd (-1.44357e-06, -0.000159827, -0.000159689, 1.35262e-06), image 4096 x 4096; SIP orders A=2, B=2, AP=2, BP=2
    With image size 4096 x 4096, central RA,Dec is (35.0502, 40.0329)
    Good region in this image (slice): None
    Reading data quality / mask file...
    DQ file: (4096, 4096) uint8 min: 0 max 0 number of pixels == 0: 16777216
    Remapping data quality / mask file...
    DQ file: (4096, 4096) uint8 min: 0 max 0 number of pixels == 0: 16777216
    Reading inverse-variance / weight file...
    Invvar map: (4096, 4096) float64 min: 0.14326538430832528 max 3384.7038778866395 median 0.3371938087998787 number of pixels == 0: 0 , number >0: 16777216
    Reading image file...
    Image pixels: (4096, 4096) float64 min: 0.0 max 75933.77972738532 median 36.92694015056645
    Running fix_saturation...
    Image pixels: (4096, 4096) float64 min: 0.0 max 75933.77972738532 median 36.92694015056645
    Invvar map: (4096, 4096) float64 min: 0.14326538430832528 max 3384.7038778866395 median 0.3371938087998787 number of pixels == 0: 0 , number >0: 16777216
    DQ file: (4096, 4096) uint8 min: 0 max 0 number of pixels == 0: 16777216
    Calling estimate_sig1()...
    Got sig1 = 0.17221072853446207
    Calling remap_invvar...
    Blanking out 0 image pixels with invvar=0
    Image pixels: (4096, 4096) float64 min: 0.0 max 75933.77972738532 median 36.92694015056645
    Invvar map: (4096, 4096) float64 min: 0.14326538430832528 max 3384.7038778866395 median 0.3371938087998787 number of pixels == 0: 0 , number >0: 16777216
    DQ file: (4096, 4096) uint8 min: 0 max 0 number of pixels == 0: 16777216
    Scaling weight(invvar) and image pixels...
    Image pixels: (4096, 4096) float64 min: 0.0 max 75933.77972738532 median 36.92694015056645
    Invvar map: (4096, 4096) float64 min: 0.14326538430832528 max 3384.7038778866395 median 0.3371938087998787 number of pixels == 0: 0 , number >0: 16777216
    Estimating sky level...
    Getting nominal zeropoint for band "NB_E"
    Traceback (most recent call last):
      File "legacypipe/new-camera-setup.py", line 254, in <module>
        main()
      File "legacypipe/new-camera-setup.py", line 244, in main
        zp0 = img.nominal_zeropoint(img.band)
      File "/global/homes/d/dstn/legacypipe/py/legacypipe/image.py", line 280, in nominal_zeropoint
        return self.zp0[band]
    AttributeError: 'WiroImage' object has no attribute 'zp0'

where we can see that the WCS header is getting read, and then the
image, data quality / flags / masks, and invvar / weight map files.  A
number of remapping steps are applied (for example, to take out an
exposure-time scaling), followed by estimation of the per-pixel noise
level ("sig1"), and the average background (sky) level.  This fails on
fetching the nominal zeropoint for the image.  We must add a dict.
For now, we'll plug in a fake value, but for production, we would
typically select the zeropoint from a clear, photometric night, with
exposures near zenith.  We add to the ``WiroImage`` class a new class
variable ``zp0``::

    class WiroImage(LegacySurveyImage):
        zp0 = dict(
            NB_E = 25.0,
        )

Of course we'll have to add the other bands eventually, but for now, onward!  Next, we get::

    > python -u legacypipe/new-camera-setup.py --camera wiro --survey-dir wiro-dir wiro/20221031/a144_zbf.fit
    ...
    Estimating sky level...
    Getting nominal zeropoint for band "NB_E"
    Got nominal zeropoint for band NB_E : 25.0
    Sky level: 36.85 count/pix
    Sky brightness: 22.401 mag/arcsec^2 (assuming nominal zeropoint)
    Does a zeropoint already exist in the image headers?  zpt= None
    Fetching Pan-STARRS stars inside this image...
    Found 5151 good PS1 stars
    Found 5151 PS1 stars in this image
    Cutting PS1 stars...
    4254 PS1 stars passed cut to be used for calibration
    Converting PS1 mags to the new camera...
    Traceback (most recent call last):
      File "legacypipe/new-camera-setup.py", line 280, in <module>
        main()
      File "legacypipe/new-camera-setup.py", line 274, in main
        phot.legacy_survey_mag = img.photometric_calibrator_to_observed(name, phot)
      File "/global/homes/d/dstn/legacypipe/py/legacypipe/image.py", line 311, in photometric_calibrator_to_observed
        colorterm = self.colorterm_ps1_to_observed(cat.median, self.band)
      File "/global/homes/d/dstn/legacypipe/py/legacypipe/image.py", line 323, in colorterm_ps1_to_observed
        raise RuntimeError('Not implemented: generic colorterm_ps1_to_observed')
    RuntimeError: Not implemented: generic colorterm_ps1_to_observed

The standard photometric calibration proceeds by fetching Pan-STARRS
PS1 stars, which get cut down to a sample used for photometric
calibration, and then transformed via a color term into the filter of
this image.  We will override the ``colorterm_ps1_to_observed()``
function, but, because WIRO is using narrow-band filters, we'll also
need to tell the code which PS-1 band should be used as the primary
band (to which the color term would get applied) for the calibration.
Let's assume we will eventually use polynomial color terms as we do
for DECam and other cameras.

::

    class WiroImage(LegacySurveyImage):
        # ...
        def get_ps1_band(self):
            from legacypipe.ps1cat import ps1cat
            # Returns the integer index of the band in Pan-STARRS1 to use for an
            # image in filter self.band.
            # eg, g=0, r=1, i=2, z=3, Y=4
            # A known filter?
            if self.band in ps1cat.ps1band:
                return ps1cat.ps1band[self.band]
            # Narrow-band filters -- calibrate againts PS1 g band
            return dict(
                NB_A = 0,
                NB_B = 0,
                NB_C = 0,
                NB_D = 0,
                NB_E = 0,
                NB_F = 0,
                )[self.band]
    
        def colorterm_ps1_to_observed(self, cat, band):
            from legacypipe.ps1cat import ps1cat
            # See, eg, ps1cat.py's ps1_to_decam.
            # "cat" is a table of PS1 stars;
            # Grab the g-i color:
            g_index = ps1cat.ps1band['g']
            i_index = ps1cat.ps1band['i']
            gmag = cat[:,g_index]
            imag = cat[:,i_index]
            gi = gmag - imag
    
            coeffs = dict(
                g = [ 0. ],
                NB_A = [ 0. ],
                NB_B = [ 0. ],
                NB_C = [ 0. ],
                NB_D = [ 0. ],
                NB_E = [ 0. ],
                NB_F = [ 0. ],
                )[band]
            colorterm = np.zeros(len(gi))
            for power,coeff in enumerate(coeffs):
                colorterm += coeff * gi**power
            return colorterm


            
At this point, the ``new-camera-setup.py`` script runs to completion.

It's time to move to the ``legacy_zeropoints.py`` script itself!
Running the command below, you'll see it first run Source Extractor
and then PsfEx to generate the PSF model file.  It will also generate
a sky background model file.  It will then proceed to look up
Pan-STARRS1 and Gaia stars for astrometric and photometric
calibration, placing these stars into the image and then fitting their
positions and brightnesses.  Eventually, it will crash with::

    > python -u legacyzpts/legacy_zeropoints.py --camera wiro --survey-dir wiro-dir --image wiro/20221031/a144_zbf.fit -v
    ...
    Fitting positions & fluxes of 1262 stars
    Off image for 25 stars
    Got photometry results for 1237 reference stars
    RA, Dec offsets (arcsec): -0.0536, -0.1868
    RA, Dec stddev  (arcsec): 2.4081, 2.0804
    RA, Dec RMS     (arcsec): 1.9789, 1.8536
    Flux S/N min/median/max: -338.1 / 138.3 / 10235409.5
    Zeropoint: using 746 good stars
    Zeropoint: using 608 stars after sigma-clipping
    Traceback (most recent call last):
      File "legacyzpts/legacy_zeropoints.py", line 1525, in <module>
        main()
      File "legacyzpts/legacy_zeropoints.py", line 827, in main
        runit(F.imgfn, F.photomfn, F.annfn, mp, **measureargs)
      File "legacyzpts/legacy_zeropoints.py", line 522, in runit
        results = measure_image(imgfn, mp, survey=survey,
      File "legacyzpts/legacy_zeropoints.py", line 356, in measure_image
        rtns = mp.map(run_one_ext, [(img, ext, survey, splinesky,
      File "/global/homes/d/dstn/astrometry/astrometry/util/multiproc.py", line 90, in map
        return list(map(f, args))
      File "legacyzpts/legacy_zeropoints.py", line 460, in run_one_ext
        return run_zeropoints(img, splinesky=splinesky, sdss_photom=sdss_photom)
      File "legacyzpts/legacy_zeropoints.py", line 1254, in run_zeropoints
        kext = imobj.extinction(imobj.band)
      File "/global/homes/d/dstn/legacypipe/py/legacypipe/image.py", line 283, in extinction
        return self.k_ext[band]
    AttributeError: 'WiroImage' object has no attribute 'k_ext'

The ``k_ext`` here is an atmospheric extinction coefficient, used to
estimate the sky transparency for our image, given its zeropoint and
airmass.  We can add to the ``WiroImage`` class::

    class WiroImage(LegacySurveyImage):
        # ...
        k_ext = dict(
            NB_A = 0.173,
        )













    


    


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

